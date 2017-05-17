'''
The config class
customized data members to be added. 
'''

class TrainConfig(object):
  def __init__(self, config_file):
    with open(config_file, 'r') as f:
      for line in f.readlines():
        tokens = line.rstrip().split()
        if tokens[0] == 'learning_rate':
          self.learning_rate = float(tokens[1])
        #elif tokens[0] == 'Dir':
        #  self.dir = tokens[1]
        #elif tokens[0] == 'Model':
        #  self.model_name = tokens[1]
        #elif tokens[0] == 'Hidden_neuron_num':
        #  self.nn_hidden_num = int(tokens[1])
        elif tokens[0] == 'batch_size':
          self.batch_size = int(tokens[1])
        elif tokens[0] == 'keep_prob':
          self.keep_prob = float(tokens[1])
        elif tokens[0] == 'num_epochs':
          self.num_epochs = int(tokens[1])
        elif tokens[0] == 'eval_every':
          self.eval_every = int(tokens[1])

  def show_config(self):
    print('=============================================================')
    print('                    Training Config Info.                    ')
    print('=============================================================')
    #print('Dir                           ', self.dir)
    #print('Model                         ', self.model_name)
    #print('Model_loc                     ', self.model_loc)
    print('learning_rate                 ', self.learning_rate)
    print('keep_prob                     ', self.keep_prob)
    print('batch_size                    ', self.batch_size)
    print('num_epochs                    ', self.num_epochs)
    print('eval_every                    ', self.eval_every)
    #print ('')

class TestConfig(object):
  def __init__(self, config_file):
    #self.model_loc = 'models/my_model'
 
    with open(config_file, 'r') as f:
      for line in f.readlines():
        tokens = line.rstrip().split()
        if tokens[0] == 'Batch_size':
          self.batch_size = int(tokens[1])
        if tokens[0] == 'Dir':
          self.dir = tokens[1]

  def show_config(self):
    print('=============================================================')
    print('                    Test config info.                        ')
    print('=============================================================')
    #print('Dir                           ', self.dir)
    #print('Batch_size                    ', self.batch_size)
    #print('Model_loc                     ', self.model_loc)
