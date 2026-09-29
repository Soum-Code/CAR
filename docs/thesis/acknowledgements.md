# Acknowledgements

<!--
  Complete as it stands: no bracketed slots, nothing that needs filling before
  submission.

  It deliberately names no one beyond the supervisor, because inventing a
  specific debt is worse than a general one honestly stated. If a particular
  person earned a line -- someone who read a draft, ran an argument at you
  until it broke, or kept a GPU session alive -- add them. A named thanks is
  worth more than a paragraph of general ones, and only you know who.
-->

This work was carried out under the guidance of Ponsuresh Manoharan, Subject
Matter Expert in Cyber Security at L&T EduTech, Chennai. I am grateful for his
supervision, and in particular for the room to follow the measurements where
they led. A project that sets out to build a system and ends up reporting why
that system cannot be built needs a supervisor willing to accept a negative
result as a result, and I had one.

I thank the School of Computer Engineering at the Kalinga Institute of
Industrial Technology for the facilities and the academic environment in which
this work was done.

I am grateful to my friends and classmates, who heard a great deal more about
step-level verification than any of them signed up for, and whose questions
were more often than not sharper than they intended.

To my family, for support that was steady and largely unremarked on: thank you.

This thesis measures other people's artifacts, and should say so. The
Math-Shepherd corpus released by Wang et al. supplies the 93,129 step-level
labels that make the central measurement possible at all; without a dataset
carrying both local and global correctness on the same steps, the gap this
thesis is about would have stayed a thought experiment. GSM8K comes from Cobbe
et al., and its inline calculator annotations are what make local validity
checkable with no model and no judge. The generators are open weights from
Mistral AI, the Qwen team and Meta.

Every GPU measurement in Chapters 5 through 8 ran on Kaggle's free allocation.
Three of those runs took six to eight hours each. A student without access to
that allocation could not have produced this thesis, and it is worth recording
that the work was gated on donated compute rather than on ideas.
