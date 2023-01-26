from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tracks

racer = tracks.Racer()


################# HYPERPARAMETERS #################
n_states=5
n_actions=2 #accelerate and steer


max_iters=500000
#Discount factor: 0 values short-term, 1 values long-term rewards
gamma=0.99
# Target network update factor, for policy bootstrapping
tau = 0.005
# Learning Rate
l_rate=1e-3

#Data buffer parameters
buf_length = 50000
batch_size = 100

#Entropy to max (less chance to converge to locals)
target_entropy= -tf.constant(n_actions,dtype=tf.float32)
log_alpha = tf.Variable(0.0,dtype=tf.float32)
alpha = tfp.util.DeferredTensor(log_alpha,tf.exp)


#Do we want to train?
training = False

#Weight params
load_weights = True
save_weights = False
actor_w_file = "around_the_world273/weights/actor"
critic1_w_file = "around_the_world273/weights/critic1"
critic2_w_file = "around_the_world273/weights/critic2"



class Get_actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense_acc1 = layers.Dense(64, activation = "relu")
        self.dense_acc2 = layers.Dense(64,activation = "relu")
        self.dense_acc3 = layers.Dense(32)
        self.dense_turn1 = layers.Dense(128, activation = "relu")
        self.dense_turn2 = layers.Dense(128,activation = "relu")
        self.dense_turn3 = layers.Dense(64)
        self.mu = layers.Dense(n_actions)
        self.ls = layers.Dense(n_actions)

    def call(self,inputs):
        acc1 = self.dense_acc1(inputs)
        acc2 = self.dense_acc2(acc1)
        acc3 = self.dense_acc3(acc2)

        turn1 = self.dense_turn1(inputs)
        turn2 = self.dense_turn2(turn1)
        turn3 = self.dense_turn3(turn2)
        joint = tf.concat([turn3,acc3],axis=1)
        mu = self.mu(joint)
        log_sigma = self.ls(joint)
        sigma = tf.exp(log_sigma)

        dist = tfp.distributions.Normal(mu,sigma)
        action = mu+ sigma * tfp.distributions.Normal(0,1).sample(n_actions)
        valid_action = tf.tanh(action)

        log_p = dist.log_prob(action)
        #Correct log_p after squashing action
        log_p = log_p - tf.reduce_sum(tf.math.log(1 - valid_action**2 + 1e-16), axis=1, keepdims=True)

        if len(log_p.shape)>1:
            log_p = tf.reduce_sum(log_p,1)
        else:
                log_p = tf.reduce_sum(log_p)
        log_p = tf.reshape(log_p,(-1,1))

        eval_action = tf.tanh(mu)

        return eval_action, valid_action, log_p
    
    @property
    def trainable_variables(self):
        return self.dense_acc1.trainable_variables + self.dense_acc2.trainable_variables + self.dense_acc3.trainable_variables + \
        self.dense_turn1.trainable_variables+ self.dense_turn2.trainable_variables+ self.dense_turn3.trainable_variables + \
        self.mu.trainable_variables + self.ls.trainable_variables


#Critics compute q-values given state and action
class Get_critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(128, activation = "relu")
        self.dense2 = layers.Dense(128,activation = "relu")
        self.qout = layers.Dense(1)

    def call(self, inputs):
        state,action = inputs
        joint = tf.concat([state, action],axis=1)
        mid = self.dense1(joint)
        out = self.dense2(mid)
        q = self.qout(out)
        return q

    @property
    def trainable_variables(self):
        return self.dense1.trainable_variables + self.dense2.trainable_variables + self.qout.trainable_variables

#Replay buffer
class Buffer:
    def __init__(self,buf_length=100000,batch_size=100):
        #Max storable tuples
        self.buf_length = buf_length
        #Num of tuples in training
        self.batch_size = batch_size

        #Current n of tuples
        self.buf_count= 0

        #Arrays for different tuples
        self.state_buf = np.zeros((self.buf_length,n_states))
        self.action_buf = np.zeros((self.buf_length,n_actions))
        self.reward_buf = np.zeros((self.buf_length,1))
        self.done_buf = np.zeros((self.buf_length,1))
        self.next_state_buf = np.zeros((self.buf_length,n_states))


    #Records a new transition
    def record(self, tuples):
        st,act,rwd,dn,nxt = tuples

        #Restart from zero if over capacity
        i = self.buf_count % self.buf_length
        self.state_buf[i] = tf.squeeze(st)
        self.action_buf[i] = act
        self.reward_buf[i] = rwd
        self.done_buf[i] = dn
        self.next_state_buf[i] = tf.squeeze(nxt)

        self.buf_count+=1

    def sample_batch(self):
        #Range
        r_range = min(self.buf_count,self.buf_length)

        #Random sampling
        indexes= np.random.choice(r_range,self.batch_size)

        st = self.state_buf[indexes]
        act = self.action_buf[indexes]
        rwd = self.reward_buf[indexes]
        dn = self.done_buf[indexes]
        nxt = self.next_state_buf[indexes]

        return((st,act,rwd,dn,nxt))

@tf.function
def update_target(target_weights,weights,tau):
    for (a,b) in zip(target_weights,weights):
        a.assign(b*tau + a * (1-tau))

def update_weights(target_weights,weights,tau):
    return(weights*tau + target_weights * (1-tau) )



#Models

actor_model = Get_actor()
critic1_model = Get_critic()
critic2_model = Get_critic()

#Targets
target_critic1 = Get_critic()
target_critic2 = Get_critic()
target_critic1.trainable = False
target_critic2.trainable = False

if load_weights:
    target_critic1([layers.Input(shape=(n_states)),layers.Input(shape=(n_actions))])
    target_critic2([layers.Input(shape=(n_states)),layers.Input(shape=(n_actions))])
    actor_model = keras.models.load_model(actor_w_file)
    critic1_model = keras.models.load_model(critic1_w_file)
    critic2_model = keras.models.load_model(critic2_w_file)


#Initially, target weights are equal

target_critic1_weights = critic1_model.get_weights()
target_critic2_weights = critic2_model.get_weights()

target_critic1.set_weights(target_critic1_weights)
target_critic2.set_weights(target_critic2_weights)

#Trying with one, might have to multiply
actor_opt = tf.keras.optimizers.Adam(l_rate)
critic1_opt = tf.keras.optimizers.Adam(l_rate)
critic2_opt = tf.keras.optimizers.Adam(l_rate)
alpha_opt = tf.keras.optimizers.Adam(l_rate)

critic1_model.compile(optimizer=critic1_opt)
critic2_model.compile(optimizer=critic2_opt)
actor_model.compile(optimizer=actor_opt)


buffer = Buffer(buf_length,batch_size)

ep_rewards = []
avg_rewards = []


#Trying without the empty steps first
def step(action):
    n = 1
    t = np.random.randint(0,n)
    action = tf.squeeze(action)
    state, reward, done = racer.step(action)
    for i in range(t):
        if not done:
            state ,t_r, done =racer.step([0, 0])
            state ,t_r, done =racer.step(action)
            reward+=t_r
    return (state, reward, done)

@tf.function
def update_critics(states, acts, rwds, dns, nxts):
    entropy_scale = tf.convert_to_tensor(alpha)
    _, new_policy_acts, log_probs = actor_model(nxts)
    q1_t = target_critic1([nxts, new_policy_acts])
    q2_t = target_critic2([nxts, new_policy_acts])

    tcritic_v = tf.reduce_min([q1_t,q2_t],axis=0)
    n_value = tcritic_v - entropy_scale*log_probs
    q_hat = tf.stop_gradient(rwds+gamma*n_value*(1-dns))

    with tf.GradientTape(persistent=True) as tape:
        q1 = critic1_model([states,acts])
        q2 = critic2_model([states,acts])
        loss_c1 = tf.reduce_mean((q1 - q_hat)**2)
        loss_c2 = tf.reduce_mean((q2 - q_hat)**2)

    critic1_gradient = tape.gradient(loss_c1, critic1_model.trainable_variables)
    critic2_gradient = tape.gradient(loss_c2, critic2_model.trainable_variables)
    critic1_model.optimizer.apply_gradients(zip(critic1_gradient, critic1_model.trainable_variables))
    critic2_model.optimizer.apply_gradients(zip(critic2_gradient, critic2_model.trainable_variables))

@tf.function    
def update_actor(states):
    entropy_scale = tf.convert_to_tensor(alpha)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(actor_model.trainable_variables)
        _, new_policy_acts, log_probs = actor_model(states)
        q1_n = critic1_model([states, new_policy_acts])
        q2_n = critic2_model([states, new_policy_acts])                    
        critic_v = tf.reduce_min([q1_n,q2_n],axis=0)      
        actor_loss = critic_v - entropy_scale*log_probs 
        actor_loss = -tf.reduce_mean(actor_loss)
    
    actor_gradient = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_model.optimizer.apply_gradients(zip(actor_gradient, actor_model.trainable_variables))


@tf.function
def update_entropy(states):
    _, _, log_probs= actor_model(states)
    with tf.GradientTape() as tape:
        alpha_loss = tf.reduce_mean(- alpha*tf.stop_gradient(log_probs + target_entropy))
    alpha_gradient = tape.gradient(alpha_loss, [log_alpha])
    alpha_opt.apply_gradients(zip(alpha_gradient, [log_alpha]))

def train(max_iters=max_iters):
    i=0
    ep=0
    avg_speed=0
    avg_reward=0

    while i<max_iters:
        prev_state = racer.reset()
        ep_reward = 0
        avg_speed +=prev_state[4]
        done=False

        while not(done):
            i+=1
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state),0)
            _, action, _ = actor_model(tf_prev_state)
            state, reward, done = step(action)

            #Success and failure are distinguished: successful termination is stored as a normal tuple
            fail = done and len(state)<5
            buffer.record((prev_state, action, reward, fail, state))
            if not(done):
                avg_speed+= state[4]

            ep_reward+=reward

            if(buffer.buf_count>batch_size):
                states,actions,rewards,dones,newstates = buffer.sample_batch()
                states = tf.stack(tf.convert_to_tensor(states, dtype=tf.float32))
                actions = tf.stack(tf.convert_to_tensor(actions, dtype=tf.float32))
                rewards = tf.stack(tf.convert_to_tensor(rewards, dtype=tf.float32))
                dones = tf.stack(tf.convert_to_tensor(dones, dtype=tf.float32))
                newstates = tf.stack(tf.convert_to_tensor(newstates, dtype=tf.float32))
                
                update_critics(states, actions, rewards, dones, newstates)
                update_actor(states)
                update_entropy(states)
                update_target(target_critic1.variables, critic1_model.variables, tau)
                update_target(target_critic2.variables, critic2_model.variables, tau)

            prev_state = state

            if i%100 == 0:
                avg_rewards.append(avg_reward)
        
        ep_rewards.append(ep_reward)
        
        #Averages from last 40 episodes
        avg_reward = np.mean(ep_rewards[-40:])
        print("Episode {}: Iterations {}, Avg. Reward = {}, Last reward = {}. Avg. speed = {}".format(ep, i, avg_reward,ep_reward,avg_speed/i))
        print("\n")

        if ep>0 and ep%50 == 0:
            print("## Evaluating policy ##")
            tracks.metrics_run(actor_model, 10)
        ep += 1

    if max_iters > 0:
        if save_weights:
            critic1_model.save(critic1_w_file)
            critic2_model.save(critic2_w_file)
            actor_model.save(actor_w_file) 
        # Plotting Episodes versus Avg. Rewards
        plt.plot(avg_rewards)
        plt.xlabel("Training steps x100")
        plt.ylabel("Avg. Episodic Reward")
        plt.ylim(-3.5,7)
        plt.show(block=False)
        plt.pause(0.001)
        print("### SAC Training ended ###")
        print("Trained over {} steps".format(i))


if training:
    train()

print("## Evaluating policy ##")
tracks.metrics_run(actor_model, 10)
tracks.newrun([actor_model])


def get_name():
    return "(Around The World)^273"

def get_actor_model():
    return actor_model
