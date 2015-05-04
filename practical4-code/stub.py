import numpy.random as npr
import sys

from SwingyMonkey import SwingyMonkey

PDEBUG = False
CRASH_REPORT = True

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

def q_choose(s, tt) :
    if PDEBUG:
        t = clamp(s,tt)
        print '( '+str(t[0]-s.boff).ljust(4)+', '+str(t[1]-s.boff).ljust(4)+', '+str(t[2]).ljust(2)+', '+str(t[3]-s.hoffn).ljust(3)+') = [ '+'{:+.1f}'.format(q(s,tt,0))+' , '+'{:+.1f}'.format(q(s,tt,1))+' ] exploited '+('JUMP' if q(s,tt,0) < q(s,tt,1) else 'fall'),
    return q(s,tt,0) < q(s,tt,1)
    
def q_max(s, tt) :
    return max( q(s,tt,0) , q(s,tt,1) )

def q_update(s, tt0, a, tt1, reward) :
    
    if PDEBUG:
        t0 = clamp(s,tt0)
        t1 = clamp(s,tt1)
        if (reward+s.discount*q_max(s,tt1)-q(s,tt0,a))*s.alpha == 0:
            print ' ; no update'
        else:
            print ' ; ( '+str(t0[0]-s.boff).ljust(4)+', '+str(t0[1]-s.boff).ljust(4)+', '+str(t0[2]).ljust(2)+', '+str(t0[3]-s.hoffn).ljust(2)+')('+('JUMP' if a else 'fall')+') updated '+'{:+.1f}'.format(q(s,t0,a))+' -> '+'{:+.1f}'.format(q(s,t0,a)+(reward+s.discount*q_max(s,t1)-q(s,t0,a))*s.alpha)
    
    return q(s,tt0,a,
        q(s,tt0,a)+(reward+s.discount*q_max(s,tt1)-q(s,tt0,a))*s.alpha
    )

def q_prior(s, t) :
    
    ret = [0, 0]
    
    ret[0] = 1 if (6<t[0] and t[0]<21) else -1
    
    if t[3] > 6 and t[0] > -20 : # we're low, but have some time left
        # don't hop just yet
        ret[0] = 0
        ret[1] = -0.1
    
    if t[2] == s.ybins-1:
        ret[1] = -9
    if t[2] == 0:
        ret[0] = -9
    
    if t[3] < 0 and t[1] < 0 : # coming in with high negative speed, captain!
        # hop!
        ret[0] = -1
        ret[1] = 1
    
    if t[0] > 20 : # we're about to hit the top of the tree
        ret[0] = -5
        ret[1] = -4 # see if re-jumping helps
    
    return ret

class Learner:

    def __init__(self):
        #self.last_state  = None
        self.last_tstate  = None
        self.last_action = None
        self.last_reward = None
        
        self.horz_speed = None
        self.gravity = None
        
        self.alpha = 0.5
        self.discount = 1
        self.ep = 0.1
        self.bsize = 10
        self.ybins = 4
        self.hoffp = 9
        self.hoffn = 5
        
        self.boff = None
        self.bins = None
        self.ysize = None
        self.hsize = None
    
    def initparams(self, swing) :
        self.horz_speed = swing.horz_speed
        self.hsize = self.horz_speed
        self.gravity = swing.gravity
        
        self.boff = swing.screen_height//self.bsize
        self.ysize = swing.screen_height//self.ybins
        self.q = [[[[q_prior(self,(yp-self.boff,yv-self.boff,yb,yh-self.hoffn)) for yh in range(self.hoffp+self.hoffn+1)] for yb in range(self.ybins)] for yv in range(2*self.boff+1)] for yp in range(2*self.boff+1)]
    
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
                print ('explored '+('JUMP' if new_action else 'fall')).ljust(50),
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
            CRASHES[reward].append(self.last_tstate)

train_iters = 100
ep_decay = 1
test_iters = 0
learner = Learner()
train_total = test_total = 0

# make a fake monkey, to set up some physics bounds
fake_monkey = SwingyMonkey()
learner.initparams(fake_monkey)

for ii in xrange(train_iters+test_iters):
    
    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length= (1000 if PDEBUG else 1),          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)
    
    # Loop until you hit something.
    while swing.game_loop():
        pass
    
    if ii<train_iters :
        if ii % (train_iters//20) == 0:
            learner.ep *= ep_decay
        train_total += swing.score
        print 'train '+str(ii).ljust(5)+'ep '+'{:.4f}'.format(learner.ep)+' : '+str(swing.score).ljust(3)+str(train_total/(ii%1000+1.0))
    else :
        learner.ep=0
        test_total += swing.score
        print 'test '+str(ii-train_iters).ljust(5)+'ep '+'{:.4f}'.format(learner.ep)+' : '+': '+str(swing.score).ljust(3)+str(test_total/(ii%1000+1.0))
    
    # Reset the state of the learner.
    learner.reset()

def tstr(s,t) :
    return '( '+str(t[0]-s.boff).ljust(4)+', '+str(t[1]-s.boff).ljust(4)+', '+str(t[2]).ljust(2)+', '+str(t[3]-s.hoffn).ljust(3)+')'

if CRASH_REPORT :
    
    if False :
        print '\nsuccesses\n'
        for t in CRASHES[1] :
            print tstr(learner,t)
    
    if True :
        print '\ntree crashes\n'
        for t in CRASHES[-5] :
            print tstr(learner,t)
    
    if True :
        print '\nwall crashes\n'
        for t in CRASHES[-10] :
            print tstr(learner,t)
