# Import libraries
import sys
from unittest.util import strclass
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
##import seaborn as sns
import keyboard as keys
import random
import math
import Tables
import os
import time
import secrets
#import decimal as d # this is actually just implemented programmatically down below 30 lines or so v v v
# import curses

print('Calculating High Numbers..')

#Python reports Infinity at 1+e308 to 1+e309

#googol = 10**100
#googolten = googol ** 10
#hugegoogol = googol**1000
sevenbil = 7000000000
threehunmil = 10000
ncaaathletes = 500000
techworkers = 11500000
##firstlownum = 1851989974471
gamesperday = 17078688
gamesperhour = 711612

calc_mode = 'default'
##calc_mode = 'stirling'
#reminder: in python ^ is **
##value = googol

context_value = 28



flips = None #number of flips each person flips, set to None here to accept input
dist = 32 #number distribution of for array of "people" running trials
max_factorial = 203500 #C and possibly Python implementation of Decimal() tends to break around binomial calculations of this magnitude because of factorials




if __name__ == "__main__": #Just so if the code is imported like a module it doesn't ask for input
    while flips == None:

        try:
            flips = int(input("\nEnter number of coin flips (50%): "))

        except:
            flips = 25
            print("Number of flips set to 25")
    

        if flips > max_factorial:

            print("\nIt is highly recommended you choose a number below "+str(max_factorial)+" for calculation speeds ")
            i = input("Continue with "+str(flips)+"? Y,n - Default(Y):")
            if i == "n":
                flips = None
            else:
                print("Calculating Context...")
                context_value = 10**10**7
                context_string = "10**10**7"
                print("\nImporting Python implementation of Decimal " +\
                    "and changing EMAX to " + context_string)
                from _pydecimal import Decimal as d
                from _pydecimal import *
                Context(Emax=context_value)
        elif flips != None:
            break
    print ('\n')
else:
    flips = 2

if ('_pydecimal' not in sys.modules): #Simply, if we didn't load the pydecimal implementation, load the C implementation
    from decimal import Decimal as d
    from decimal import *

def d_round(n, prec): #Decimal version of round()
    global context_value
    getcontext().prec = prec
    p = n * 1
    getcontext().prec = context_value

    return p



# Probability of success for each experiment

# p = 0.5

print("(Point of reference: A p of .1 is a coin which lands tails 90% of the time. The algorithm checks for tails so this is very fast.")
print("Checking for p = .9 runs very slow and it would be better to check for .1 and swap heads for tails.)")
p_str = input("\nProbability [p=.5]")


if p_str == "":
    p_str = "0.5"
elif float(p_str) <= 0:
    p_str = ".01"
elif float(p_str) > 1: 
    p_str = ".99"
p = d(p_str)
print("p = " + p_str)


#maxzero = d_ceil(d(1 / p) ** d(flips)) 
maxzero = d_round(1/d(d(1-p) ** d(flips)), 1) #flips before achieving 0% (also, the probability of any given toss results)
##maxzero = sevenbil


d_pi = d(math.pi)
d_e = d(math.e)


##def calc_thresh(x): #Based on sample sizes of 30 for number of flips up to 100,000. >100,000 will likely shrink too quickly
##    sign = lambda x: x and (1, -1)[x<0]
##    minmax = lambda x,min_x,max_x: (1-sign(min_x-x))*(1-sign(x-max_x))
##    x = (1.6*math.log(x)+.5)/4*minmax(x,0,10) +(-.6213/40.11*(x-9.99)+2.0933)/4 * minmax(x,10, 50)  \
##        + (-.613/50*(x-50)+1.472)/4 * minmax(x,50, 100) + (-.5229/400*(x-500)+.3361)/4*minmax(x,100,500) \
##        + (-.120829375/500*(x-500)+.3361)/4 * minmax(x,500,1000) + (-.200791625/4000*(x-1000) + .215270625)/4 * minmax(x,1000,5000) \
##        + (.056427371875/5000*(x-5000)+.014479)/4 * minmax(x,5000, 10000)  \
##        + (-.054261071875/90000*(x-10000)+.070906371875)/4 * minmax(x,10000, 100000) \
##        + (6.65*math.log(x**50))/(2*x)*(1-sign(100000-x))
##
##    return d(x)




thresh = 0 #d(d(10 ** 4) * d.sqrt(d(2*d_pi))) / d(flips * d.sqrt(d(flips)))

print("\nSample Trial Distributions:\n[0/default] Logarithmic\n[1] Even\n[2] Low\n[3] User\n[4] User (Log)")

trials_input = input("\nSelect: ")
if len(trials_input) > 0: trials_input = int(trials_input)
else: trials_input = 0

high = maxzero
if high < dist: dist = int(maxzero)

if (trials_input == 0 or trials_input == None):
    #Logarithmic distribution

    trials = [int(d(i - d((i ** 2)/dist) + 10 ** (d(i ** 2/dist).log10() * d(high).log10()/d(dist).log10()))) for i in range(1, dist+1)]
    #trials[-1] = int(high)
    trials[0] = 1
elif (trials_input == 1):
    # Even Distribution up to dist
    # trials =[i for i in range(1, dist+1)]   

    ##Even Distribution up to flips / high
    high = flips #10000
    trials =[int(i * high / dist) for i in range(1, dist+1)]
    dist = len(trials) + 1
elif (trials_input == 2):
    #Original Dataset
    trials = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 50, 65, 75, 90, 100, 150, 200, 300, 400, 500, 750, 800, 900, 1000, 1200, 1500, 2000, 5000, 10000, 50000, 100000, 500000, 1000000]
    dist = len(trials) + 1
elif (trials_input == 3):
    trials = [0, int(input("\nSelect: "))]
else: 
    trials_input2 = input("\nSelect: ")
    high = int(trials_input2)
    # print(str(high))
    trials = [int(d(i - d((i ** 2)/dist) + 10 ** (d(i ** 2/dist).log10() * d(high).log10()/d(dist).log10()))) for i in range(1, dist+1)]
    trials[-1] = int(high)
    trials[0] = 1
    


trials[0] = 1







#Graphing Probability
##trials = [1]
##dist = len(trials) + 1

print("\nDistribution calculated...")


sim = input("\nSimulated? [y/N]")
if sim.upper() == "Y":
    simulated = True
else: 
    simulated = False #for commenting convenience




def remove_dupes(t): #For ascending lists, elegantly replace duplicates in a list of integers with near sized integers
    c = len(t)
    for i in range(1, c):
        if i >= 1:
            if t[i] <= t[i-1]:t[i] = t[i-1] + (t[i-1] - t[i]) + 1
    return t


#More variable declarations
trials = remove_dupes(trials)
#np.random.seed()
flips_array = [None] * len(trials)
flips_array2 = [None] * len(trials)
cs_flips_result = [None] * len(trials)

_a =    ( 1.00000000000000000000, 0.57721566490153286061, -0.65587807152025388108,
         -0.04200263503409523553, 0.16653861138229148950, -0.04219773455554433675,
         -0.00962197152787697356, 0.00721894324666309954, -0.00116516759185906511,
         -0.00021524167411495097, 0.00012805028238811619, -0.00002013485478078824,
         -0.00000125049348214267, 0.00000113302723198170, -0.00000020563384169776,
          0.00000000611609510448, 0.00000000500200764447, -0.00000000118127457049,
          0.00000000010434267117, 0.00000000000778226344, -0.00000000000369680562,
          0.00000000000051003703, -0.00000000000002058326, -0.00000000000000534812,
          0.00000000000000122678, -0.00000000000000011813, 0.00000000000000000119,
          0.00000000000000000141, -0.00000000000000000023, 0.00000000000000000002
       )



def d_gamma (x: d): #Decimal verison of gamma for 0 < x < 1
   x +=1 # Shouldn't really do this here for consistency with mathematics, but I just think it's cleaner
   one = d(1.0)
   y  = d(x) - one
   sm = _a[-1]
   for an in _a[-2::-1]:
      sm = d(d(sm) * y) + d(an)
   return d(one / sm)





def d_floor(n): #Decimal version of math.floor()
    getcontext().rounding = ROUND_FLOOR
    p = n * 1
    getcontext().rounding = ROUND_HALF_EVEN
    return p




def d_factorial(n: d):
    if (n == 0): return 1
    if (n == 2): return 2
    n2 = d(n)
    i = d(n)
    while i >= 3:
        i -= 1
##        n2 = d(n2 * i)
        try:
            n2 = d(n2 * i)
        except:
            print ("Below are the numbers we tried so hard to multiply, perhaps try increasing context?")
            print (str(n2), str(i))
            print("Error:", sys.exc_info()[0])
            exit()
    n = d(n2) * d_gamma(i-1) #For smaller numbers, to represent the decimal we have to multiply 1 < x < 2 mixed decimal by gamma at the end
    return n


def remove_zero_trail(num): #remove trailing zeros for < 1 needs rewrite for efficiency
        #if (d(num) >= 1): return num #not sure why this is even here
        n = num
        c = len(n)-1
        while n[c] == "0": 
            n = n[:c]
            c -= 1
        if n[c] == '.': return n[:c]
        return n


def remove_pythnotation(num): #Removes scientific notation up to 2000 digits
        
        b = f'{num:.2000f}'
        if num > d(b):
            print ("error removing scientific notation. exiting..")
            exit(-1)
        b = remove_zero_trail(b) #this doesn't appear to work everytime, not sure why, yet: check again
        return b




def scinotate(numpass: d, prec: int): #manual scientific notation for very large and very small numbers
    sign = '' 
    if (numpass < 0): ##value is negative so store it and add it later after string manipulation
        sign = '-'
        n = str(numpass)
        numpass = d(n[1:])

    num = numpass
    n = ""

    if (num == 0): return "0"
    elif (num >= 1) and (d_floor(d(num).log10()) <= len(str(10**prec))): return  sign + str(d_round(numpass, prec)) #mixed decimal digits less than prec
    elif(num < 1): #unmixed decimal, digits less than prec
        n = remove_zero_trail(remove_pythnotation(d(numpass)))
        if (len(n) < len(str(10**prec))): return sign + n 


    #typical cases:
    if (n == ""): n = str(numpass)

    l = len(n)
    per = n.find(".")

    if (per == -1): #no period, large numbers
        b = n[0] + '.' + n[1:prec]
        t = str(d_round(d(b), prec))
        a = t + 'e+' + str(l-1)
    else: #period, small numbers
        for c in range(per, l):
            if n[c] != "0":
                if n[c] == '.':
                    mem = c
                else:  break;
        b = num * 10**(c-mem)
        t = remove_zero_trail(str(d_round(d(b), prec)))
        a = t + 'e-' + str(len(str(10**(c-2))))

    return sign + a




def print_table(): #Print a table of lists with labels
    Column = Tables.Table.Column
    nums = list(map(str, trials))
    speeds = list(map(str, flips_array))
    speeds2 = list(map(str, flips_array2))

    #Format numbers for screen width
##    termrows, termcolumns = os.popen('stty size', 'r').read().split() #Mac only?
    termcolumns, termrows = os.get_terminal_size()
    for i in range(0, len(nums)):

        if math.floor(d(nums[i]).log10()/d(10).log10()) > int(termcolumns) - len(str(flips)) -60:
            nums[i] = scinotate(d(nums[i]), 2)
    
    desc = [ '', 'Population', 'Protection', 'Simulated']
    clear()

    print (Tables.Table(
        Column('Rounds', nums),
        Column('Results from ' + str(flips)  + ' flips', speeds, align=Column.LEFT),
        Column('Simulated', speeds2, align=Column.LEFT)))
      
    print (str(maxzero) + " trials until 0% at p=" + str(p))
    return

def percent(one, two): #return a percentage of two numbers with the sign
    percent = round(one / two, 2) * 100
    return str(int(percent)) + "%"

def clear(): #clear terminal independent of OS
    os.system('cls' if os.name == 'nt' else 'clear')
    # os.system('cls||echo -e \\\\033c')
    return


def stirling_binom(numtrials, numtosses, i, prob):
    if i < 1: return 0

    i_fac = d.sqrt(d(2 * d_pi * i))*d(d(i/d_e)**i)
    nt_fac = d.sqrt(d(2 * d_pi * numtosses))*d(d(numtosses/d_e)**numtosses)
    com_fac = d.sqrt(d(2 * d_pi * (numtosses - i)))*d(d((numtosses - i)/d_e)**(numtosses - i))
        
    return d(numtrials) * nt_fac/(i_fac*com_fac) * d(prob) ** i * (1 - d(prob)) ** d(numtosses - i)   


def gaussian_dist(numtrials, numtosses, thresh, prob): ## Normal distribution solved for x, reduced in terms of ln, e, log(e)(x), whatever nomenclature you prefer
    thresh = d(thresh)
    a = d(numtosses * prob) #mean
    b = d(a * d(1 - prob)) #variance

    return numtosses - (d(a) + d.sqrt(-2*b*d.ln((thresh*d.sqrt(2*d_pi*b))/numtrials)))

def binomial_dist(numtrials, numtosses, i, prob):
##Much more readable version
    t = d(numtrials)
    n = d(numtosses)
    x = d(i)
    p = d(prob)
    term = d(1-p)**d(n-x) #recursive with the term variable, harder to unravel, easier on the eyes
    term = d(p**x)*term
    term = d_factorial(n)/d(d_factorial(x)*d_factorial(n-x)) * term
    term = d(t*term)
    return term
##   return d(numtrials) * d(d(d(d_factorial(d(numtosses))/(d_factorial(d(i))*d_factorial(d(numtosses - d(i))))) * d(d(d(prob)**d(i)) * d(d(d(1) - d(prob)) ** d(numtosses - d(i))))))


def useful_time(i, total, time_left): # Convert a time to a readable format
    if (time_left > 60 and time_left <= 3600):
        time_left = percent(i, total) + ' ' + str(round(time_left / 60, ndigits=2)) + " minutes left"
    elif (time_left > 3600 and time_left <= 86400):
        time_left = percent(i, total) + ' ' + str(round(time_left / 3600, ndigits=2)) + " hours left"
    elif (time_left > 86400):
        time_left = percent(i, total) + ' ' + str(round(time_left / 86400, ndigits=2)) + " days left"                
    else:
        time_left = percent(i, total) + ' ' + str(round(time_left, ndigits=2)) + " seconds left"
    return time_left

def calc_binom(numtrials, numtosses, prob): #Calculate lowest probabilities > thresh for a given population, number of trials, and assumed binomial prob, update table, print it, 
    index = trials.index(numtrials)
    
    mid = d(numtosses*prob) #the p of the tosses is the highest probability in a binomial
    mid_int = int(mid)
    if (mid_int < 0): mid_int = 1
 
    #This is for 'time to calculate' later
    if (mid_int >= 2222): interval = 100
    else: interval = 2 / 125 * mid_int
    if interval <= 0: interval = .000001
    
    step = mid_int / interval
    last_step = step

    a = d(numtosses * p)
  
    #This whole section is for skipping calculating trials which will most-likely be zero probability and also determining the threshold to test against
    identity_dist = binomial_dist(1, numtosses, mid, prob) # this is the dist result if we only ran one *set* i.e. 1xnumtosses
    if (numtosses==1):
        thresh = numtrials * p
        mad = 0
    else:
        
        thresh = identity_dist
##        mad =d(d(d(d(c) / d(numtrials)) * d(.110171458167779)))
        mad = (numtosses * p * ( 1 - p)) 
##        thresh = d(d(c) / d(numtrials))
##        mad = d(0)
    #if (c <= thresh): return scinotate(d_round(c, 2), 4) + " P of 50%"
    start = 0
    if (prob >= .33): #this should probably be reworked to include number of tosses, but maybe irrelevant
        start = int(gaussian_dist(numtrials, numtosses, thresh, prob)) #start at the gaussian dist using the binomial as the threshold to save calculation time
        start = 0 if start < 0 else start
    


    low_stirling = None
    low_value = 0

    time_left_interval = time.perf_counter()

    for i in range(start, mid_int+1, 1):
        #if (maxzero == numtrials): return "0"
   
        
        if(calc_mode == 'default'): #default #This was implemented and never used, this whole if statement could just be replaced with low_value =
            low_value = binomial_dist(numtrials, numtosses, i, prob)
        elif (calc_mode == 'both'):
            low_value = binomial_dist(numtrials, numtosses, i, prob)
            low_stirling = stirling_binom(numtrials, numtosses, i, prob) 
        else:
            low_value = stirling_binom(numtrials, numtosses, i, prob)

        if (low_value >= thresh) and low_stirling == None:
           
            #i_for_low_value = d(d_round(d(d(i/(numtosses))*d(100)+d(mad/2)),3))
            i_for_low_value = d_round(d(d(i/(numtosses))*100),3)
            low_value_str = scinotate(i_for_low_value,4) + '%' #lowest value achievable by percent

            return scinotate(i, 4)+'/'+scinotate(numtosses-i,4)  + ' ('+low_value_str +')' #+','+scinotate(thresh,4)  #+ ', ' +str(scinotate(low_value,4)) #+ ', '+ scinotate(y, 4)

     
        #String formatting for time left
        if (i > last_step):

            time_left = (time.perf_counter() - time_left_interval) / step * (mid_int - i)
            time_left_interval = time.perf_counter()
            flips_array[index] = useful_time(i, mid_int, time_left)
            print_table()
            last_step += step
    #end loop

    return scinotate(d_round(identity_dist, 2), 4) + " p of " + str(prob*100) + "%" #We didn't find it so just send the highest probability of the binomial
    


def sim_binom(numtrials, numtosses, prob, suppress_table):
    lowest = numtosses
    amount =  last_step = time_left = time_left_interval = 0
    index = trials.index(numtrials)
    rand = random.SystemRandom()
    interval = .05
    step = interval * numtrials
    tic = time.perf_counter()
    
    for person in range(numtrials):
        amount = 0
        for flip in range(numtosses):
                res = float(-1)
                #res = rand.random()
                #res = secrets.randbelow(2)
  
                # while res < 0 or res > 1:
                #     res = secrets.randbelow(4)-1

                # while res < 0:
                #     res = secrets.randbelow(3)-1

                res = secrets.randbelow(100) / 100
                   
                if (res <= prob): amount = amount + 1
        if (amount < lowest):
            lowest = amount
            if (lowest == 0): return 0

        #String formatting for time left
        if (person > last_step and not suppress_table):
            if (last_step == step):
                time_left_interval = time.perf_counter() - tic
                time_left = time_left_interval * ((1 / interval) - 1)
            else:
                time_left -= time_left_interval
                flips_array2[index] = useful_time(person, numtrials, time_left)
            print_table()
            #print (interval * (trials[i]*2 for i in range(person, len(trials))))
            last_step += step

    return lowest


def d_maddev(lst): # MAD (mean absolute deviation)
    lst2 = []
    lst2 = np.empty(len(lst), dtype=Decimal)
    avg = d(d_sum(lst) / len(lst))
    for i in range(0, len(lst)):
        lst2[i] = d(abs(d(lst[i] - avg)))
        
    if lst2 is not None:
        avg = d(d_sum(lst2) / d(len(lst2)))
        #maddev = d_round(avg, 3)
        maddev = avg
    else:
        maddev = 0

    return maddev

def stddev(lst): # STD (standard deviation)
    lst2 = []
    lst2 = np.empty(len(lst), dtype=float)
    avg = sum(lst) / len(lst)
    for h in range(0, len(lst)):
        lst2[h] = (lst[h] - avg) ** 2
    if lst2 is not None:
        avg = sum(lst2) / len(lst2)
        stddev = round(math.sqrt(avg), 3)
    else:
        stddev = 0
    del lst2
 
    return stddev

def findfloat(lst, value) -> int:
    abslst = lst
    for i in range(0,len(lst)):
        abslst[i] = abs(lst[i] - value)
    return np.argmin(abslst)

def d_sum(lst):
    total = 0
    for index in lst:
        total += index
    return total

def d_mean(lst):
    return d(d_sum(lst) / len(lst))

def d_mode(lst, value):
    return findfloat(lst, value)

def print_d_as_f(lst, prec):
    for i in range(0,len(lst)):
        print(float(d_round(lst[i], prec)))
    return

def probability_chart():

    mx = 200
    prec = 2

    x = []
    y = []

    z = 1000 #trials
##    z = 1

    flip_distribution = 42 #<-log
    if flips < flip_distribution: flip_distribution = int(flips)
    if flip_distribution < 2: flip_distribution = 2
    start = 0
#Logarithmic dist
    flip_list = [int(d(i+start - d(((i+start) ** 2)/flip_distribution) + 10 ** (d((i+start) ** 2/flip_distribution).log10() * d(flips).log10()/d(flip_distribution).log10()))) for i in range(1, flip_distribution+1)]

### Linear dist
##    start = int(input("\nEnter number of coin flips to start (50%): "))
##    step = int(input("\nEnter number of coin flips to step (50%): "))
####    start = d(30000)

##    flip_distribution = flips #<-linear
##    flip_list = [i for i in range(start, flips, step)]
##
##    flip_list[-1] = int(flips)

    flip_list[0] = 1
    flip_list = remove_dupes(flip_list)

##    flip_list = [4999,5000]
    
    lst_means_x = []
    lst_means_y = []
    lst_modes_y = []
    lst_mad_y = []
    

    row = 1
    num = 0


    for g in flip_list:
        mid = d(d(g)/2)
        if (mid < 0): mid = 0
        num += 1
        c = binomial_dist(z, g, mid, p)
        for i in range(1,mx+1):
            tic = time.perf_counter()
            result_sim = sim_binom(row, g, p, True)
            f =binomial_dist(z, g, result_sim, p)
            x.append(c)
            y.append(f)
            time_left = (time.perf_counter() - tic)*(mx - i)
            time_left = useful_time(i, mx, time_left)
            print(str(num) +'('+str(g)+')'+ ' of ' + str(len(flip_list)) + '\n')
            print('\t'+str(i) + ' of ' + str(mx) + ' with ' + time_left + ' .\n')

        clear()
            
        x_subset = x[-1] # we take the last few values of x and y (in magnitude of mx for y)
        y_subset = y[-mx:]

        lst_means_x.append(x_subset)
        lst_means_y.append(d_mean(y_subset))

        lst_modes_y.append( y[ -mx + d_mode( y_subset, lst_means_y[-1]) ] )
        lst_mad_y.append(d_maddev(y_subset))


    clear()

    print("Flips Dataset")
    print_d_as_f(flip_list, 3)
    print("Binomial X Values")
    print_d_as_f(lst_means_x, 3)
    print("Mean Y Values")
    print_d_as_f(lst_means_y, 3)
    print("MAD Y Values")
    print_d_as_f(lst_mad_y, 3)

 
    fig, (ax1) = plt.subplots()


    ax1.scatter(x,y, alpha=0.04) # Results
    ax1.plot(lst_means_x, lst_means_y, '.-') #means
    ax1.plot(lst_means_x, lst_modes_y, '.-') #modes

    for n in range(0, len(lst_means_x)):
        comp_array_x = []
        comp_array_x.append(lst_means_x[n])
        comp_array_x.append(lst_means_x[n])
        comp_array_y = []
        comp_array_y.append(lst_means_y[n] - lst_mad_y[n] / 2)
        comp_array_y.append(lst_means_y[n] +lst_mad_y[n] / 2)
        
        ax1.plot(comp_array_x, comp_array_y, 'g--')

    
    ax1.grid(True)
    ax1.set_title('Scatter')

##    lt.legend((p1[0], p2[0]), ('Men', 'Women'))

    #fig.tight_layout()
    
    plt.show()
    return



try:
    maxzero = d(math.ceil(float(maxzero)))
    maxzero = scinotate(d(remove_pythnotation(maxzero)), 20) #last chance to clean up our target value to be nicer looking and appropriate size relative to screen width

except:
    maxzero = scinotate(d(remove_pythnotation(maxzero)), 20) #last chance to clean up our target value to be nicer looking and appropriate size relative to screen width




if (__name__ == "__main__"): #Again, if we import code don't calculate tables
    rerun = True
    while (rerun):
        num = 0
        for row in trials:
            flips_result = calc_binom(row, flips, p)
            flips_array[num] = flips_result
            print_table()


            if simulated:
                flips_result_sim = sim_binom(row, flips, p, False)
                cs_flips_result[num] = str(d_round(binomial_dist(row, flips, flips_result_sim, p), 4)) #This is relevant after closing the probability chart
                #flips_array2[num] = str(percent(flips_result_sim, flips))+', '+ str(flips_result_sim) + ', ' + str(d_round(binomial_dist(row, flips, flips_result_sim, p), 4))
                # flips_array2[num] = str(flips_result_sim) +'/'+str(flips-flips_result_sim)+' ('+str(percent(flips_result_sim, flips))+')'
                flip_ratio = round(flips_result_sim / flips * 100, 2)
                flip_ratio = 100 - flip_ratio if flip_ratio > 50 else flip_ratio 
                flips_array2[num] = str(flips_result_sim) +'/'+str(flips-flips_result_sim)+' ('+ str(flip_ratio) + '%)'
                print_table()

            num += 1
        rerun = False
        print ("Finished for p=" + str(p) + " of " + str(flips) + "!")


        if len(trials) > 3:
            laymans = input("\nTranslate three random row results in english? [y/N]")

            if laymans.upper() != "N":

                # random_choices = random.sample(flips_array, 3)

                # print(random_choices)
                
                nums = list(map(str, trials))
        
            #Format numbers for screen width
            
                termcolumns, termrows = os.get_terminal_size()
                for i in range(0, len(nums)):

                    if math.floor(d(nums[i]).log10()/d(10).log10()) > int(termcolumns) - len(str(flips)) -60:
                        nums[i] = scinotate(d(nums[i]), 2)

                num_trials = len(flips_array)
                while True:
                    random_indices = random.sample(range(num_trials), 3)
                    if 0 not in random_indices:
                        break

                # Process each chosen index
                for index in random_indices:
                    choice = flips_array[index]
                    parts = choice.split(' ')
                    num_tails, num_heads = map(float, parts[0].split('/'))
                    
                    num_tails = int(num_tails)
                    num_heads = int(num_heads)
                    total_flips = num_tails + num_heads

                    print(f"\n Out of {nums[index]} rounds of people tossing coins {total_flips} times: The lowest possible number of tails I could achieve is {num_tails} for at least one round, with {num_heads} heads. "
                        f"Or we could say the same in reverse, that I could achieve only {num_tails} heads out of {total_flips} with all the rest ({num_heads}) being tails.")
            
            if simulated:
                rerun = input("\nRe-run? [Y/n]")
                if rerun.upper() == "N":
                        rerun = False
                else: 
                        rerun = True #for commenting convenienceupper(input("\nRe-run?[y/N]"))



    #Plot the results as a scatterplot

##    probability_chart()

##    if simulated:
##        for num in range(0, len(trials)):
##            print(cs_flips_result[num])

    ##for s in stddev_list:
    ##    print (s)

    #print ('\a') #beep

        #keys.wait(" ")
    
