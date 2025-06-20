# CS-4780-5780-Assignment-5-HMM-Statistical-Learning-Theory-solution

Download Here: [CS 4780/5780 Assignment 5: HMM &amp; Statistical Learning Theory solution](https://jarviscodinghub.com/assignment/assignment-5-hmm-statistical-learning-theory-solution/)

For Custom/Original Work email jarviscodinghub@gmail.com/whatsapp +1(541)423-7793

Problem 1: Viterbi Algorithm [50 points]
We learned Hidden Markov Models (HMMs) as a generative model, and the Viterbi
Algorithm is used to compute the most likely configuration of the hidden variables.
HMMs are useful for various tasks such as Speech Recognition, Machine Translation, and
Signal Encoding, and consequently the Viterbi algorithm is highly important technique
in Machine Learning. In this assignment, we will investigate a code translation problem
that can also be solved by the Viterbi algorithm.
In the beginning of the semester, Ian created a simple linguistic code that utilizes several Greek alphabets in order to safely encrypt his password. After a few months, he
forgets the original English password, and only remembers the Greek encryption. He
believes that strong similarity between English and Greek alphabets was incorporated
for his encryption, whereas there was no deterministic mapping between English and
Greek characters. Thus we hypothesize that sequences of Greek characters correspond
to English words and plan to build an HMM to help him decipher his password.
3-1
Formally speaking, we represent a Greek word as x =(x1, x2…xT ), where xt ∈ {α, τ, η, γ, ω}
is the t
th character in the word (represented in Greek script). Given each Greek word, our
goal is to predict the most probable English word y=(y1, y2…yT ), where yt ∈ {a, i, p, s} is
the English translation of the t
th character (represented in English script). The following
tables give the transition and emission probabilities of our HMM.
a i p s
a 0.05 0.1 0.15 0.7
i 0.1 0.05 0.25 0.6
p 0.45 0.15 0.05 0.35
s 0.35 0.2 0.15 0.3
START 0.1 0.4 0.2 0.3
Table 1: Transition Probabilities P(yt
|yt−1) where yt−1 on each row, yt on each column
α τ η γ ω
a 0.4 0.2 0.1 0.2 0.1
i 0.3 0.1 0.4 0.1 0.1
p 0.1 0.1 0.1 0.2 0.5
s 0.1 0.4 0.1 0.3 0.1
Table 2: Emission Probabilities P(xt
|yt) where yt on each row, xt on each column
In HMMs, the transition probability P(yt
|yt−1) decides the probability of the current
English character given the previous English character, whereas the emission probability P(xt
|yt) determines the probability of a Greek translation given a English character.
Based on these two model probabilities, we can compute the most likely English translation of each Greek word x =(x1, x2…xT ) via the following HMM formula:
argmaxy1,y2…yT
P(y1, y2…yT |x1, x2…xT ) = argmaxy1,y2…yT
P(y1)P(x1|y1)
Y
T
t=2
P(xt
|yt)P(yt
|yt−1)
Using the above equation and the Viterbi algorithm, answer the following questions:
(a) [5 points] Let δs,t be the probability of the most probable English sequence corresponding to the first t Greek observations (x1, x2, …, xt) that end with the English
character s. Write the recurrence relation of δs,t in terms of model probabilities1
and specify the initial conditions2
for 2D dynamic programming (s ∈ {a, i, p, s},
1 ≤ t ≤ T)
1 We have three probabilities: transition, emission, and start probabilities given in the Table 1 & 2
2 To perform dynamic programming, you have to initialize the base cases at t = 1 for all s ∈ {a, i, p, s}
3-2
(b) [25 points] Ian’s encrypted password is ”ωαγτ ητ γαω”. What are the most likely
English translations for each of the Greek encrypted words: 1) ητ , 2) ωηα, 3) ωαγτ?
For each Greek word, all you have to do is to fill the 2D dynamic programming table3
whose entries are δs,t with specifying the backtracking path that corresponds to the
most probable translation. What is indeed Ian’s English password?
(c) [10 points] Suppose English has m characters and Greek has n characters in their
alphabets. What is the running time of the translation from a Greek word of length
T into an English word via the Viterbi algorithm? Compare it to the running time
of the brute-force algorithm naively considering all possible combinations in terms
of big-O complexity. (You could assume reading the model probabilities takes only
a constant time)
(d) [5 points] Given the transition and emission probabilities of a first-order HMM
model, describe how to compute the probability of a Greek word x=(x1, x2…xT ).
(i.e., P(X = (x1, x2, …, xt))) You have to clearly specify the equations you formulate
for calculating this probability as well as a precise 2-4 sentence description of how
your algorithm would work. (note: Solutions with exponential time complexity will
get the full credits, but a clear description of sub-exponential time algorithm will get
the bonus points)
(e) [5 points] We can similarly utilize HMMs to translate sentences of one language into
another. In this sentence-level translation, each character in our problem corresponds
to a word, and a sequence of characters corresponds to a sentence. How well would
this HMM-based approach perform as compared to the state-of-the-art (like Google
Translate) for languages common today, say English and Spanish? Briefly explain
why you think so in 2-3 sentences.
Problem 2: Statistical Learning Theory [50 points]
For the questions below, clearly explain all steps in your derivations. You may use any
result proved in class.
(a) Find the tightest lower bound that you can on the VC dimension for each of the
following hypotheses classes. Present the sets of points that can be shattered for
each of the following:
• Intervals of the reals For an instance space in R, let H be the set of all
intervals on the real line.
• Pairs of intervals of the reals For an instance space in R, let H be the set
of all pairs of intervals on the real line.
3 Conventionally s will vary on the row side and t will vary on the column side in the table.
3-3
• Rectangular hypotheses centered at the origin For an instance space in
R
2
, let h ∈ H be a hypothesis: h(x) = 1{−a < x1 < a, −b < x2 < b} for
a, b ∈ R.
• Rectangular hypotheses centered arbitrarily For an instance space in R
2
,
let h ∈ H be a hypothesis: h(x) = 1{a < x1 < b, c < x2 < d} for a, b, c, d ∈ R.
• Circular hypotheses centered arbitrarily For an instance space in R
2
, let
h ∈ H be a hypothesis: h(x) = 1{||x − c||2 < r} for r ∈ R, c ∈ R2
(b) Let {0, 1}
n be the instance space of a binary classification task. This means that an
instance x consists of n binary-valued features. Let Hn be the class of all boolean
functions over the input domain. What is |Hn| and the VC dimension of Hn? Show
all work leading you to your answer.
(c) Now, suppose we are interested in a binary classification task, where our examples
x are all 100-dimensional binary vectors of the form x = (x1, x2, . . . , x100), where
each xi ∈ {0, 1}. As above, we want to learn a binary hypothesis h over the instances. Consider a “restricted” linear hypothesis h that can be described using a
100-dimensional weight vector w = (w1, w2, . . . , w100) and a bias b, where all components of the weight vector are binary, i.e. ∀i : wi ∈ {−1, 1} and the bias in an
integer between -15 and 15: b ∈ {−15, −14, . . . , 0, . . . , 14, 15}. The classification rule
for classifying an example x is sign(wT x + b).
The hypothesis space H consists of all such w, b. Given a training set S with |S| = n
examples, and a hypothesis with 0 training error, give a bound for the prediction
error of this hypothesis that will hold with probability 1 − δ in terms of δ and n.
(d) For H defined as above, let hˆ
S = argminh∈HErrS(h). How large must the training
set size |S| be for P

|ErrS(hˆ
S) − ErrP (hˆ
S)| ≥ 0.1

≤ 0.1?

