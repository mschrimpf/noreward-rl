import glob
import os

import argparse
import tensorflow as tf
from tensorflow.python.summary import event_accumulator as ea

from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns
sns.set(style="darkgrid")
sns.set_context("paper")

def plot(params):
  ''' beautify tf log
      Use better library (seaborn) to plot tf event file'''

  log_path = params['logdir']
  all_workers = glob.glob(os.path.join(log_path, 'train_*'))

  smooth_space = int(params['smooth'])
  color_code = params['color']
  tag = params['var']
  maxstep = params['maxstep']

  x_raw = []
  y_raw = []

  for worker in all_workers:
    print(worker)
    acc = ea.EventAccumulator(worker)
    acc.Reload()

    for s in acc.Scalars(tag):
      if not maxstep or s.step < maxstep:
        x_raw.append(s.step)
        y_raw.append(s.value)
  
  sorted_xy = sorted(zip(x_raw, y_raw), key=lambda xy: xy[0])
  x_raw, y_raw = zip(*sorted_xy)

  # smooth curve
  x_smooth = []
  y_smooth = []
    
  for i in range(0, len(x_raw), smooth_space):
    if i + 2*smooth_space < len(x_raw):
      x_smooth.append(x_raw[i])
      y_smooth.append(sum(y_raw[i:i+smooth_space]) / float(smooth_space))    
    elif i + smooth_space < len(x_raw):
      x_smooth.append(x_raw[i])
      y_smooth.append(sum(y_raw[i:]) / float(len(x_raw) - i))    

  x_raw = [x*1e-6 for x in x_raw]
  x_smooth = [x*1e-6 for x in x_smooth]
  plt.figure()
  plt.subplot(111)
  plt.title(params['title'], fontsize=16)  
  plt.plot(x_raw, y_raw, color=colors.to_rgba(color_code, alpha=0.4))
  plt.plot(x_smooth, y_smooth, color=color_code, linewidth=1.5)
  plt.xlabel('Number of Global Training Steps (in millions)', fontsize=14)
  plt.ylabel('Intrinsic Reward' if params['intrinsic'] else 'Episode Reward', fontsize=14)
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
  plt.gca().xaxis.offsetText.set_fontsize(12)
  plt.savefig(params['filename'])


  plt.show()


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', default='./logdir', type=str, help='logdir to event file')
  parser.add_argument('--var', default='global/episode_reward', type=str, help='which scalar to plot')
  parser.add_argument('--smooth', default=100, type=float, help='window size for average smoothing')
  parser.add_argument('--color', default='#4169E1', type=str, help='HTML code for the figure')
  parser.add_argument('--title', default='Rewards vs Global Step', type=str)
  parser.add_argument('--filename', default='fig.png', type=str)
  parser.add_argument('--intrinsic', type=bool)
  parser.add_argument('--maxstep', default=None, type=int)

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  plot(params)
