"""The CAR control loop.

One pass over a question:

    plan -> for each target:
        generate step + uncertainty features
        draw exploration coin      (BEFORE the gate -- propensity must be known)
        score -> gate -> CONTINUE / VERIFY / ABSTAIN
        if VERIFY: call an external verifier, maybe revise, spend budget
        feed the observed outcome back to the calibrator
    -> final answer

The ordering constraint that matters: verification happens BEFORE a step is
committed as trusted state, not at the end of the chain. That is the whole
point -- an error caught at step 1 costs one call, the same error caught at
step 5 has already contaminated four downstream conclusions.
"""

from __future__ import annotations

import time

from car.control.budget import Budget
from car.control.gate import ControlGate
from car.tracing.trace import TraceWriter
from car.types import Decision, Example, StepRecord, Trajectory, Verdict


class CARAgent:
    """Composes generator, scorer, calibrator, gate and verifier into one policy.

    Every component is injected so that baselines are the same loop with pieces
    swapped, not parallel implementations that could drift apart. An
    apples-to-apples baseline comparison depends on this.
    """

    def __init__(
        self,
        *,
        generator,
        scorer,
        calibrator,
        gate: ControlGate | None = None,
        verifier=None,
        budget_per_question: int = 3,
        max_steps: int = 8,
        trace: TraceWriter | None = None,
    ) -> None:
        self.generator = generator
        self.scorer = scorer
        self.calibrator = calibrator
        self.gate = gate or ControlGate()
        self.verifier = verifier
        self.budget_per_question = budget_per_question
        self.max_steps = max_steps
        self.trace = trace

    def run(self, example: Example) -> Trajectory:
        budget = Budget(self.budget_per_question)
        traj = Trajectory(
            example_id=example.example_id,
            question=example.question,
            gold_answer=example.gold_answer,
        )

        plan = self.generator.plan(example)[: self.max_steps]
        accepted: list = []

        for i, target in enumerate(plan):
            t0 = time.perf_counter()
            gen = self.generator.step(example, target, accepted)
            score = self.scorer.score(gen.features)

            # Exploration coin first, and independent of the score. Reversing
            # this order would make the observed labels' propensity unknown and
            # silently invalidate the adaptive calibration.
            force_explore = (
                self.calibrator.should_explore()
                if hasattr(self.calibrator, "should_explore")
                else False
            )

            # Descendants are not yet generated, so influence is approximated
            # by how many planned steps remain downstream of this one.
            remaining = max(0, len(plan) - i - 1)

            outcome = self.gate.decide(
                score,
                self.calibrator.threshold,
                n_descendants=remaining,
                budget=budget,
                force_explore=force_explore,
            )

            verdict: Verdict | None = None
            revised = False
            observed_error: int | None = None

            if outcome.decision == Decision.VERIFY and self.verifier is not None:
                if budget.spend(1, exploration=outcome.forced_exploration):
                    result = self.verifier.verify(gen.step, example.question)
                    verdict = result.verdict
                    traj.verification_calls += 1

                    if verdict == Verdict.CONTRADICTED:
                        observed_error = 1
                        if result.revised_claim:
                            gen.step.claim = result.revised_claim
                            revised = True
                    elif verdict == Verdict.SUPPORTED:
                        observed_error = 0
                    # INSUFFICIENT leaves observed_error as None: we genuinely
                    # learned nothing, and recording it as 0 would be a lie the
                    # calibrator would then act on.

            if outcome.decision == Decision.ABSTAIN:
                traj.aborted = True

            record = StepRecord(
                step=gen.step,
                features=gen.features,
                score=score,
                threshold=outcome.threshold,
                influence=outcome.influence,
                gate_value=outcome.gate_value,
                decision=outcome.decision,
                forced_exploration=outcome.forced_exploration,
                verdict=verdict,
                revised=revised,
                label=gen.true_label,
                budget_remaining=budget.remaining,
                latency_s=time.perf_counter() - t0,
            )
            traj.steps.append(record)

            # Close the loop. `was_accepted` -- not `decision == CONTINUE` --
            # is what marks a label as belonging to the accept region, which is
            # the region whose risk we are controlling.
            if observed_error is not None and hasattr(self.calibrator, "update"):
                self.calibrator.update(
                    observed_error,
                    was_exploration=outcome.forced_exploration,
                    was_accepted=ControlGate.was_accepted(outcome),
                )

            if outcome.decision == Decision.ABSTAIN:
                break
            accepted.append(gen.step)

        traj.final_answer = self._finalise(accepted)
        traj.correct = self._score_answer(traj, example)

        if self.trace is not None:
            self.trace.write(traj)
        return traj

    def run_all(self, examples: list[Example]) -> list[Trajectory]:
        return [self.run(ex) for ex in examples]

    @staticmethod
    def _finalise(accepted: list) -> str | None:
        return accepted[-1].claim if accepted else None

    @staticmethod
    def _score_answer(traj: Trajectory, example: Example) -> bool | None:
        """Final-answer correctness.

        In simulation, a trajectory is correct when no wrong step survived
        unverified -- the propagation assumption made explicit. On real
        datasets this is replaced by exact-match or execution against the gold
        answer.
        """
        labels = [r.label for r in traj.steps if r.label is not None]
        if not labels:
            return None
        survived_wrong = any(
            r.label is False and r.decision == Decision.CONTINUE for r in traj.steps
        )
        return not survived_wrong
