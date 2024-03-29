from utils.env_wrapper import PendulumWrapper, LunarLanderContinuousWrapper, BipedalWalkerWrapper
from env_kyon import SimStudent
class train_params:
    
    # Environment parameters
    ENV = 'kyon'                     # Environment to use (must have low dimensional state space (i.e. not image) and continuous action space)
    RENDER = False                          # Whether or not to display the environment on the screen during training
    RANDOM_SEED = 99999999                  # Random seed for reproducability
    NUM_AGENTS = 1                    # Number of distributed agents to run simultaneously
    
    # ======================= WARINGING ========================
    REMAIN_DATABASE = False
    
    # Create dummy environment to get all environment params
    if ENV == 'Pendulum-v0':
        dummy_env = PendulumWrapper()
    elif ENV == 'LunarLanderContinuous-v2':
        dummy_env = LunarLanderContinuousWrapper()
    elif ENV == 'BipedalWalker-v2':
        dummy_env = BipedalWalkerWrapper()
    elif ENV == 'BipedalWalkerHardcore-v2':
        dummy_env = BipedalWalkerWrapper(hardcore=True)
    elif ENV == 'kyon':
        dummy_env = SimStudent()
    else: 
        raise Exception('Chosen environment does not have an environment wrapper defined. Please choose an environment with an environment wrapper defined, or create a wrapper for this environment in utils.env_wrapper.py')
     
    STATE_DIMS = dummy_env.get_state_dims()
    STATE_BOUND_LOW, STATE_BOUND_HIGH = dummy_env.get_state_bounds()
    ACTION_DIMS = dummy_env.get_action_dims()
    ACTION_BOUND_LOW, ACTION_BOUND_HIGH = dummy_env.get_action_bounds()
    V_MIN = dummy_env.v_min
    V_MAX = dummy_env.v_max
    del dummy_env
    
    # Training parameters
    NUM_STEPS_TRAIN = 10000      # Number of steps to train for
    MAX_EP_LENGTH = 1000           # Maximum number of steps per episode
    REPLAY_MEM_SIZE = 10000      # Soft maximum capacity of replay memory
    NOISE_SCALE = 0.45               # Scaling to apply to Gaussian noise
    NOISE_DECAY = 0.9999            # Decay noise throughout training by scaling by noise_decay**training_step
    DISCOUNT_RATE = 0.99            # Discount rate (gamma) for future rewards
    N_STEP_RETURNS = 3              # Number of future steps to collect experiences for N-step returns
    UPDATE_AGENT_EP = 10            # Agent gets latest parameters from learner every update_agent_ep episodes
    
    # Network parameters
    CRITIC_LEARNING_RATE = 0.0001
    ACTOR_LEARNING_RATE = 0.0001
    CRITIC_L2_LAMBDA = 0.0          # Coefficient for L2 weight regularisation in critic - if 0, no regularisation is performed
    DENSE1_SIZE = 400               # Size of first hidden layer in networks
    DENSE2_SIZE = 300               # Size of second hidden layer in networks
    FINAL_LAYER_INIT = 0.003        # Initialise networks' final layer weights in range +/-final_layer_init
    NUM_ATOMS = 51                  # Number of atoms in output layer of distributional critic
    TAU = 0.001                     # Parameter for soft target network updates
    USE_BATCH_NORM = False          # Whether or not to use batch normalisation in the networks
  
    # Files/Directories
    SAVE_CKPT_STEP = 100                # Save checkpoint every save_ckpt_step training steps
    CKPT_DIR = './ckpts/' + ENV             # Directory for saving/loading checkpoints
    CKPT_FILE = 'kyon.ckpt-100000'                        # Checkpoint file to load and resume training from (if None, train from scratch)
    LOG_DIR = './logs/train/' + ENV         # Directory for saving Tensorboard logs (if None, do not save logs)
    
    
class test_params:
   
    # Environment parameters
    ENV = train_params.ENV                                  # Environment to use (must have low dimensional state space (i.e. not image) and continuous action space)
    RENDER = False                                          # Whether or not to display the environment on the screen during testing
    RANDOM_SEED = 999999                                    # Random seed for reproducability
    
    # Testing parameters
    NUM_EPS_TEST = 100                                      # Number of episodes to test for
    MAX_EP_LENGTH = 10000                                   # Maximum number of steps per episode
    
    # Files/directories
    CKPT_DIR = './ckpts/' + ENV                             # Directory for saving/loading checkpoints
    CKPT_FILE = './ckpts/kyon/kyon.ckpt-1000000'                                       # Checkpoint file to load and test (if None, load latest ckpt)
    RESULTS_DIR = './test_results'                          # Directory for saving txt file of results (if None, do not save results)
    LOG_DIR = './logs/test/' + ENV                          # Directory for saving Tensorboard logs (if None, do not save logs)


class play_params:
   
    # Environment parameters
    ENV = train_params.ENV                                  # Environment to use (must have low dimensional state space (i.e. not image) and continuous action space)
    RANDOM_SEED = 999999                                    # Random seed for reproducability
    
    # Play parameters
    NUM_EPS_PLAY = 5                                        # Number of episodes to play for
    MAX_EP_LENGTH = 10000                                   # Maximum number of steps per episode
    
    # Files/directories
    CKPT_DIR = './ckpts/' + ENV                             # Directory for saving/loading checkpoints
    CKPT_FILE = None                                        # Checkpoint file to load and run (if None, load latest ckpt)
    RECORD_DIR = './video'                                  # Directory to store recorded gif of gameplay (if None, do not record)


    
