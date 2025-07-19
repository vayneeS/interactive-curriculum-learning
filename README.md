# Interactive_Curriculum_Learning
Analysis of data from experiment to study how an adaptive MAB algorithm compares to baseline algorithms for learning a toy motor task, consisting of driving a cursor across a channel of variable width in a given time.
 
3 conditions: 
Curriculum Learning (CL):
Tasks were scheduled because (i) LP was highest or lowest (exploitation) or pseudorandomly (exploration). A MAB algorithm from Clement et al. 2013 was adapted to choose a task every 4 trials. 
  
Random: 
Blocks of 7 tasks are grouped and shuffled to obtain a sequence of tasks (intermediate random difficulty as opposed to shuffling all blocks, to get closer to the design of Choi et al (2008) Fix condition)  

Error Adaptation (EA): 
Adaptation of Choi et al’s Adaptr condition. The challenge point is determined during trials at the start of practice. Number of trials per task varies, with a minimum of 7.15% and a maximum of 36.3%. Consecutive blocks were limited by reshuffling.

Experiment proceeded in several stages over 2 days:
Day 1:
Pre-test - Trials to measure skills at beginning of experiment
Training - Running experiment with respective conditions
Post-test - Trials to measure skills after training
Day 2:
Transfer - Trials to measure skills on a related task where the diameter of the channel was changed
Retention - Trials to measure skills on same task as post-test 

More details in publication : 
Sungeelee, Vaynee, et al. "Interactive curriculum learning increases and homogenizes motor smoothness." Scientific Reports 14.1 (2024): 2843.
