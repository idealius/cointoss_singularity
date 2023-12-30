# Hello World

This program was conceived, researched, and somewhat haphazardly programmed just before the pandemic in early 2020. Aside from adding command line options, I have not continued development on it because other repositories and the usual, but I wish to when time is available.

# What is it?

A college level mathematics teacher will tell you to calculate the number of trials for a cointoss before we get all heads or all tails, we simply do 2^(trials). This is the reduction of the binomial distribution formula.

The most concise explanation I can give is:

**So, if we flip 5 coins (trials), it will take 2^5 i.e. 32 total cointosses, whether we assign to multiple people, or done by one to achieve a streak of all 5 heads or all 5 tails.**

***Yet, there is still a nonzero possibilty of getting all heads/tails (or close to) when looking at 2^(trials - x) where x is the gap before we reach 2^(trials). So, if we cycle through x from a logarithmic or linear series of numbers this program will show those nonzero probabilities, and their level.***

## But, how do we calculate that?

A statistican would have some great ideas. I am not one, I am a math hobbyist that is just interested in this particular corner of academics, and this is my result. 

# Inspiration

If you have a roomful of people performing a cointoss, what are the expected breakdowns of "streaks" to occur respective to not only the number of cointosses, but the number of rooms performing the trial?

If you'd like to understand what algorithm underlies the code, and why I chose it, please message me and we can chat, but it's in fact very basic. The accuracy is not 100%, but it seems to be scalable so I think it's most likely on the correct track.

## Improvements?

The first thing on my mind is getting to run with pypy for faster runtimes and simulations. But, we could work on QRNG sources. Right now it is set to a RNG that is for security encryption, with the idea that it could be more "random." (And, it does appear to be that way, from my experimentation.)

## The Point?

When the population of a probability exceeds a certain amount the gaussian distribution isn't as seemingly accurate anymore because the long tails of a distribution suddenly allow extended runs of wins / losses. In the past this was fine because computers couldn't calculate those tails timely due to factorial equations being difficult to solve for, but now computers have advanced enough to accomodate complex factorial equations to some extent.
