#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#from matplotlib.animation import FuncAnimation
import os
import time 
from .env_config import env_config
from datetime import timedelta

        
class LearnFrame():
    def __init__(self, model, data, initialze = False):
        self.model=model
        self.data = data

        try:
            os.mkdir(env_config.checkpoint_dir)
        except:
            print("Checkpoint directory already exists")

        if initialze:
            self.optimize()
            self.test()

    def scheduler_step(self):
        self.model.scheduler_step()

    def reset(self):
        self.data.reset()

    def collect(self):
        self.model.train()
        done, reward = self.data.collect(self.model)
        return done

    def train(self,test_freq, save_freq, steps,test_steps):
        # measure time as well and print elapsed time and estimated time
        start_time = time.time()
        path = env_config.checkpoint_dir + "/" 
        for i in range(self.model.optimizer._optimizer_steps_counter, steps):

            self.optimize()

            if (i+1)%test_freq == 0:
                self.test(steps=test_steps)
            
            if (i+1) % save_freq == 0:
                dire = path + f"{(i+1)}"
                
                try:
                    os.mkdir(dire)
                except:
                    pass
                self.save(dire)
            
            self.print_progress(i, steps, start_time)
        dire = path + "final"
        try:
            os.mkdir(dire)
        except:
            pass
        self.save(dire)
        env_config.xai.writer.close()

    def optimize(self):
        """
        Gets training data from data, and trains the algorithms one step. 
        Parameters
        ----------

        Returns
        -------
        loss
            Loss objects (shape depends on the algorithm)
        """        
        self.model.train()
        optimizer_return = False

        while optimizer_return == False:
            #Get the next batch
            X = self.data.train_data()
            if X is None:
                return 
            
            optimizer_return = self.model.optimize(X)
        
    def test(self, steps = 1):
        self.model.eval()
        test_return = False
        
        for _ in range(steps):
            while test_return == False:
                X = self.data.test_data()
                test_return = self.model.test(X)
                
    def save(self, file_name, save_data = True):
        self.data.save(file_name)
        if save_data:
            self.model.save(file_name)

    def load(self, file_name, load_data = True):
        self.model = self.model.load(file_name)
        if load_data:
            self.data.load(file_name = file_name)


    def get_anim(self, name = None, interval = 100):
        frames = []
        self.data.reset()
        self.model.eval()
        frames.append(self.data.env.render())
        done = False
        while(not done):
            done, reward = self.data.collect(self.model)
            frames.append(self.data.env.render())
 

        fig, ax = plt.subplots()
        def animate(t):
            ax.cla()
            ax.imshow(frames[t])

        anim = FuncAnimation(fig, animate, frames=len(frames), interval = interval)

        if name is None:
            return anim
        fig.suptitle(name, fontsize=14) 
          
        # saving to m4 using ffmpeg writer 
        writervideo = animation.FFMpegWriter(fps=60) 
        anim.save(name + ".mp4", writer=writervideo) 
        plt.close()

    def print_progress(self, step, total_steps, start_time):
        ratio = step / total_steps
        percent = ratio * 100
        
        # Build progress bar of width 10 (like tqdm)
        bar_length = 10
        filled_length = int(bar_length * ratio)
        
        bar = self.build_bar(ratio, bar_length)
        
        # Elapsed time
        elapsed = time.time() - start_time
        elapsed_str = self.convert_time(elapsed)

        
        
        # Speed calculation (steps per second)
        speed = step / elapsed if elapsed > 0 else 0
        speed_str = f"{speed:,.2f}"

        step_str = f"{step:0{len(str(total_steps))}d}"

        #remaining time
        remaining = (total_steps - step) / speed if speed > 0 else 0
        remaining_str = self.convert_time(remaining)
        
        # Format speed with 2 decimals and commas for thousands
        speed_str = f"{speed:,.2f}"

        # Construct the progress string like tqdm
        progress_str = (f"\r{percent:.2f}%|{bar}| {step_str}/{total_steps} "
                        f"[{elapsed_str}<{remaining_str}, {speed_str}it/s]")
        
        print(progress_str, end='\r', flush=True)

        # Print a new line at the end
        if step == total_steps - 1:
            print()

    def build_bar(self,ratio, bar_length=20):
        filled = int(ratio * bar_length)
        empty = bar_length - filled

        black = '\033[30m'  # Black (visible only on light backgrounds)
        gray = '\033[90m'   # Bright black (gray)
        reset = '\033[0m'

        return f"{black}{'█' * filled}{gray}{'░' * empty}{reset}"

    def convert_time(self, seconds):
        """
        Converts seconds to a string in the format HH:MM:SS
        """
        td =  timedelta(seconds=seconds)
        total_hours = td.days * 24 + td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        seconds = td.seconds % 60

        formatted = f"{total_hours}:{minutes:02}:{seconds:02}"
        return formatted  # → "27:46:40"