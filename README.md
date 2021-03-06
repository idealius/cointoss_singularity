# Hello World

This program was conceived, researched, and somewhat haphazardly programmed just before the pandemic in early 2020. Aside from adding command line options, I have not continued development on it because other repositories and the usual, but I wish to when time is available.

# What is it?

A college level mathematics teacher will tell you to calculate the number of trials for a cointoss before we get all heads or all tails, we simply do 2^(trials). This is the reduction of the binomial distribution formula.

The most concise explanation I can give is:

**So, if we flip 5 coins (trials), it will take 2^5 i.e. 32 total cointosses, whether we assign to multiple people, or done by one to achieve a streak of all 5 heads or all 5 tails.**

***Yet, there is still a nonzero possibilty of getting all heads/tails (or close to) when looking at 2^(trials - x) where x is the gap before we reach 2^(trials).***

## But, how do we calculate that?

A statistican would have some great ideas. I am not one, I am a math hobbyist that is just interested in this particular corner of academics, and this is my result. 

# Inspiration

I was playing a very popular online game a year before I started this project. And, at that time, despite the probabilities for win/loss of this particular game being .49/.51 (almost a cointoss) I noticed even after improving sometimes I would lose 20 games in a row. This was very strange to me and I wanted to understand how this could even be possible. I would call these "vortices" or "the vortex" because it felt as though the probability of this happening couldn't be chance, and yet the source of my losses was unknown, "So... multiple... *things* are causing me to lose unfairly??" (<- Hence, vortex) But, it turns out, chance can cause this! So, I looked up the number of concurrent players (at the time) and it was around 7-8 million people. Then, I attempted to learn college-level statistics to understand how these streaks can occur, and I began with quickly coded simulations in Python, that I would paste in Excel, but eventually I dropped Excel and stayed with advancing the Python program. 

Disclaimer: I taught myself to code in High School back when dial-up was a thing, so if you see some faux pas, a thousand apologies.

If you'd like to understand what algorithm underlies the code, and why I chose it, please message me and we can chat, but it's in fact very basic. The accuracy is not 100%, but it seems to be scalable so I think it's most likely on the correct track.

## Improvements?

The first thing on my mind is getting to run with pypy for faster runtimes and simulations. But, we could work on QRNG sources. Right now it is set to a RNG that is for security encryption, with the idea that it could be more "random." (And, it does appear to be that way, from my experimentation.)

## The Point?

I think it can raise existential questions, um it can show how pointless it is to be an online gamer. Different stuff like that.
