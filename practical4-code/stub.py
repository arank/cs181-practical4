import numpy.random as npr
import sys

from SwingyMonkey import SwingyMonkey

import time
import pickle
import os

PDEBUG = False
FFDEBUG = True
CRASH_REPORT = True
LEARN_REPORT = True

ALPHA    = 0.5
GAMMA    = 1
EP       = 1
EP_DECAY = 0.99
B_SIZE   = 10
Y_BINS   = 8
H_OFF_P  = 9
H_OFF_N  = 5

TRAIN_ITERS = 1000
TEST_ITERS  = 100
RECORD_TRAIN_CRASHES = False
IN_TEST_MODE = False

POST_FMT = 'pvyh' # mutate this if you change statespace dimensions
USE_POST = 'prior_0_0'

if CRASH_REPORT :
    CRASHES = {1:[],-5:[],-10:[]}

def clamp(s,tt) :
    return (
        max(0,min(s.boff, tt[0]//s.bsize )+s.boff),
        max(0,min(s.boff, tt[1]//s.bsize )+s.boff),
        max(0,min(s.ybins-1, tt[2]//s.ysize )),
        max(0,min(s.hoffp, tt[3]//s.hsize )+s.hoffn)
    )

def q(s, tt, a, val=None) :
    t = clamp(s,tt)
    if val is not None : 
        s.q[t[0]][t[1]][t[2]][t[3]][a] = val
    return s.q[t[0]][t[1]][t[2]][t[3]][a]

def q_hit(s, tt, a) :
    t = clamp(s,tt)
    s.hits_given[t[0]][t[1]][t[2]][t[3]][a] += 1

def q_choose(s, tt) :
    if PDEBUG:
        t = clamp(s,tt)
        print '( '+str(t[0]-s.boff).ljust(4)+', '+str(t[1]-s.boff).ljust(4)+', '+str(t[2]).ljust(2)+', '+str(t[3]-s.hoffn).ljust(3)+') = [ '+'{:+.1f}'.format(q(s,tt,0))+' , '+'{:+.1f}'.format(q(s,tt,1))+' ] exploited '+('JUMP' if q(s,tt,0) < q(s,tt,1) else 'fall'),
    return q(s,tt,0) < q(s,tt,1)
    
def q_max(s, tt) :
    return max( q(s,tt,0) , q(s,tt,1) )

def q_update(s, tt0, a, tt1, reward) :
    if LEARN_REPORT :
        q_hit(s,tt0,a)
    
    if PDEBUG :
        t0 = clamp(s,tt0)
        t1 = clamp(s,tt1)
        if (reward+s.discount*q_max(s,tt1)-q(s,tt0,a))*s.alpha == 0:
            print ' ; no update'
        else:
            print ' ; ( '+str(t0[0]-s.boff).ljust(4)+', '+str(t0[1]-s.boff).ljust(4)+', '+str(t0[2]).ljust(2)+', '+str(t0[3]-s.hoffn).ljust(2)+')('+('JUMP' if a else 'fall')+') updated '+'{:+.1f}'.format(q(s,tt0,a))+' -> '+'{:+.1f}'.format(q(s,tt0,a)+(reward+s.discount*q_max(s,tt1)-q(s,tt0,a))*s.alpha)
    
    return q(s,tt0,a,
        q(s,tt0,a)+(reward+s.discount*q_max(s,tt1)-q(s,tt0,a))*s.alpha
    )

def q_export_posterior(slug, posterior) :
    print 'writing '+POST_FMT+'_'+slug+'...'
    pickle.dump(posterior, open( POST_FMT+'_'+slug+'.p' ,'wb'))
    print 'done.'

def q_import_prior(slug) :
    if os.path.isfile(POST_FMT+'_'+slug+'.p') :
        print 'loading '+POST_FMT+'_'+slug+'...'
        return pickle.load(open( POST_FMT+'_'+slug+'.p' ,'rb'))
    else :
        print 'could not find '+POST_FMT+'_'+slug+'.'
        return None

# increment this if you tweak prior parameters
PRIOR_VSN = '0_0'

def q_prior(s, t) :
    
    ret = [0, 0]
    
    ret[0] = 1 if ( 6<t[0] and t[0]<18 ) else -1
    ret[1] = 0 if ( t[0]<18 ) else -2 # or worse, but we'll get to that
    
    if t[3] > 4 and t[0] > -15 : # we're low, but have some time left
        # don't hop just yet
        ret[0] = 0
        ret[1] = -0.1
    
    if t[2] == s.ybins-2:
        ret[1] = -7
    if t[2] == s.ybins-1:
        ret[1] = -9
    if t[2] == 0:
        ret[0] = -9
    if t[2] == 1:
        ret[0] = -7
    
    if ( t[3] <= 0 ) and ( t[0]-(t[1]*(t[3]-1))-(s.gravity*(t[3]*(t[3]+1.))//s.bsize) < 7 ) : # coming in with high negative speed, captain!
        # hop!
        ret[0] = -1
        ret[1] = 1
    
    if t[0] >= 20 : # we're about to hit the top of the tree
        ret[0] = -9.9
        ret[1] = -9.1 # see if re-jumping helps
    
    return ret

def tstr(s,tt) :
    t = clamp(s,tt)
    return '( '+str(t[0]-s.boff).ljust(4)+', '+str(t[1]-s.boff).ljust(4)+', '+str(t[2]).ljust(2)+', '+str(t[3]-s.hoffn).ljust(3)+')'

REPORT_WIDTH = 25

def report_tree_crash(s,t,a, newline) :
    print tstr(s,t),
    print ' <'+('JUMP' if a else 'fall')+'>',
    if t[3]//s.hsize == 0 :
        if t[0] < 56 :
            print ' leading--LOW'.ljust(REPORT_WIDTH),
        elif t[0] > 200 :
            print ' leading--high'.ljust(REPORT_WIDTH),
        else :
            print ' leading--??'.ljust(REPORT_WIDTH),
    elif t[3] < 0 :
        if t[1] < 0 :
            print ' TRAILING--LOW'.ljust(REPORT_WIDTH),
        elif t[0] > 0 :
            print ' TRAILING--high'.ljust(REPORT_WIDTH),
        else :
            print ' TRAILING--??'.ljust(REPORT_WIDTH),
    else :
        print ' NOT YET REACHED TREE?'.ljust(REPORT_WIDTH),
    
    if newline :
        print ''

def report_wall_crash(s,t,a, newline) :
    print tstr(learner,t),
    print ' <'+('JUMP' if a else 'fall')+'>',
    if t[2] > 300 :
        print ' high'.ljust(REPORT_WIDTH),
    elif t[2] < 100 :
        print ' LOW'.ljust(REPORT_WIDTH),
    else :
        print ' ??'.ljust(REPORT_WIDTH),
    
    if newline :
        print ''

class Learner:

    def __init__(self):
        #self.last_state  = None
        self.last_tstate  = None
        self.last_action = None
        self.last_reward = None
        
        self.horz_speed = None
        self.gravity = None
        
        self.alpha = ALPHA
        self.discount = GAMMA
        self.ep = EP
        self.bsize = B_SIZE
        self.ybins = Y_BINS
        self.hoffp = H_OFF_P
        self.hoffn = H_OFF_N
        
        self.boff = None
        self.bins = None
        self.ysize = None
        self.hsize = None
        
        self.q = None
        self.hits_given = None
    
    def initparams(self, swing) :
        self.horz_speed = swing.horz_speed
        self.hsize = self.horz_speed
        self.gravity = swing.gravity
        
        self.boff = swing.screen_height//self.bsize
        self.ysize = swing.screen_height//self.ybins
        
        self.q = q_import_prior(USE_POST)
        if self.q is None :
            print 'generating fresh '+POST_FMT+'_'+PRIOR_VSN
            self.q = [[[[q_prior(self,(yp-self.boff,yv-self.boff,yb,yh-self.hoffn)) for yh in range(self.hoffp+self.hoffn+1)] for yb in range(self.ybins)] for yv in range(2*self.boff+1)] for yp in range(2*self.boff+1)]
            q_export_posterior('prior_'+PRIOR_VSN,self.q)
        if LEARN_REPORT :
            self.hits_given = [[[[[0,0] for yh in range(self.hoffp+self.hoffn+1)] for yb in range(self.ybins)] for yv in range(2*self.boff+1)] for yp in range(2*self.boff+1)]
    
    def reset(self):
        #self.last_state  = None
        self.last_tstate  = None
        self.last_action = None
        self.last_reward = None
    
    def get_future(self, state):
        # get time until tree is reached
        time = state['tree']['dist']/self.horz_speed
        
        # get position at time
        future_pos = -0.5*self.gravity*time**2 + state['monkey']['vel']*time + state['monkey']['top'] - state['tree']['bot']
        
        # get velocity at time
        future_vel = -self.gravity*time + state['monkey']['vel']
        
        #return {'vel': future_vel, 'pos': future_pos}
        return (int(future_pos), int(future_vel), state['monkey']['top'], state['tree']['dist'])
    
    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''
        
        tstate = self.get_future(state)
        
        if npr.rand() < self.ep :
            # go exploring
            new_action = npr.rand() < 0.1
            if PDEBUG:
                print ('explored '+('JUMP' if new_action else 'fall')).ljust(55),
        else :
            # do the right thing
            new_action = q_choose(self, tstate)
        
        # You might do some learning here based on the current state and the last state.
        
        if self.last_tstate is not None:
            q_update(self, self.last_tstate, self.last_action, tstate, self.last_reward)
        
        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.
        
        #new_state  = state
        
        self.last_action = new_action
        #self.last_state  = new_state
        self.last_tstate = tstate
        
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        
        self.last_reward = reward
        
        if CRASH_REPORT and reward :
            if RECORD_TRAIN_CRASHES or IN_TEST_MODE :
                CRASHES[reward].append((self.last_tstate,self.last_action))
            
            if reward == -5 :
                report_tree_crash(self,self.last_tstate,self.last_action,False)
                if PDEBUG :
                    time.sleep(6)
                
            if reward == -10 :
                report_wall_crash(self,self.last_tstate,self.last_action,False)
                if PDEBUG :
                    time.sleep(6)

learner = Learner()
train_total = test_total = 0

# make a fake monkey, to set up some physics bounds
fake_monkey = SwingyMonkey()
learner.initparams(fake_monkey)

for ii in xrange(TRAIN_ITERS+TEST_ITERS):
    
    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length= ((80 if FFDEBUG else 1000) if PDEBUG else 1),          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)
    
    # Loop until you hit something.
    while swing.game_loop():
        pass
    
    if ii<TRAIN_ITERS :
        learner.ep *= EP_DECAY
        train_total += swing.score
        print 'train '+str(ii).ljust(5)+'ep '+'{:.4f}'.format(learner.ep)+' : '+str(swing.score).ljust(3)+str(train_total/(ii+1.))
    else :
        IN_TEST_MODE = True
        learner.ep=0
        test_total += swing.score
        print 'test '+str(ii-TRAIN_ITERS).ljust(5)+': '+str(swing.score).ljust(3)+str(test_total/(ii-TRAIN_ITERS+1.))
    
    # Reset the state of the learner.
    learner.reset()

print 'train avg: '+str(1.0*train_total/(TRAIN_ITERS))+' ('+str(TRAIN_ITERS)+')'
print 'train avg: '+str(1.0*test_total/(TEST_ITERS))+' ('+str(TEST_ITERS)+')'

q_export_posterior(str(TRAIN_ITERS)+'_'+str(time.time()), learner.q)

if CRASH_REPORT :
    
    if False :
        print '\nsuccesses\n'
        for report in CRASHES[1] :
            tt = report[0]
            a = report[1]
            print tstr(learner,t)
    
    if True :
        print '\ntree crashes\n'
        for report in CRASHES[-5] :
            report_tree_crash(learner,report[0],report[1],True)
    
    if True :
        print '\nwall crashes\n'
        for report in CRASHES[-10] :
            report_wall_crash(learner,report[0],report[1],True)
