CODE_OUTPUT_TIP = f'''
The output should include two components:  
1. **Reward Function Class**  
2. **Reward Scale Configuration**  

**Formatting Requirements:**  
- The reward function class should be provided as a Python code string:
  ```python  
  # Your reward function class here  
  ```  
- The reward scale configuration should be provided as a YAML code string:
  ```yaml  
  # Your reward scale configuration here  
  ```  

**Guidelines:**  
- Use only the environment's predefined variables; any new tensor or variable must be on the same device as the input tensors.  
- Refer to the given `reward_function` and `reward_scale` templates for guidance.  
- The reward function consists of two types of terms:  
  1. **Command Tracking Terms:** Crucial terms directly evaluated by the user to ensure task success.  
  2. **Regularization Terms:** Penalization terms to suppress undesired behaviors, ensuring stable learning. 
  All rewards except regularization terms is critically related to the final performance.  
- The provided reward scale configuration has been tested in locomotion tasks and can serve as a reference. 
- If you don't want one term of reward, just set the scale to 0.

**Most Important**: 
You need to return the same reward function as template. Only reward scale is needed to tune. 
You must output the complete code of reward function, no omissions is allowed for reward function.
To train a stable policy at least, it's highly recommended to explore from the given reward scales. 
'''

INITIAL_SYSTEM = f'''
You are a **reward engineer** designing reward functions for locomotion based on reinforcement learning. 
Your goal is to create a reward function that optimally guides the agent in learning the locomotion task.  

**Structure of the Reward Function:**  
- **Python Class:** Defines the reward function.  
- **YAML Configuration:** Specifies the scaling factors for different reward terms.  

#### **Example: Reward Function Class**  
```python
{{reward_class}}
```

#### **Example: Reward Scale Configuration**  
```yaml
{{reward_config}}
```

**Key Notes:**  
- The reward configuration defines **scaling factors** for different reward terms.  
- The return of each reward function is positive. If you want to penalize it, you should only change its scale to negative.
- Reward scale names must match their corresponding reward function terms.  
- Final reward = **Sum of all reward terms after applying the scale factors**.  

Refer to the following guidelines while writing your reward function:  
{CODE_OUTPUT_TIP}  
'''

INITIAL_USER = '''
Task: Design a reward function that enables the robot to walk effectively. The function should ensure the robot achieves stable locomotion without falling while following the given commands—linear velocity (lin_vel), angular velocity (ang_vel), and the desired gait.

Evaluation Criteria:

    Survivability - The robot should remain upright and avoid falling.
    Command Adherence - The robot should accurately follow the given lin_vel and ang_vel.
    Stability - The robot should maintain a steady and controlled motion.

Additional Notes:

    If following the desired gait proves too difficult, prioritize training the robot to follow the velocity commands.
    The desired gait is automatically computed within the environment.
    Task note: {task_note}
'''

JUDGE_SYSTEM = f'''
You are an evaluator assigned to select the optimal reward parameter set for a robot locomotion task using reinforcement learning. You will be provided with evaluation data containing the follwing metrics:
1. Termination Rate: The count of failures observed during evaluation, where lower values indicate better performance.
2. Other Metrics: Numerical values ranging from 0 to 1, where higher values represent better outcomes.

Your task is to carefully analyze both metrics and determine the reward parameter set that demonstrates the best overall performance. Provide the identification number corresponding to the highest-performing set based on the evaluation data.
'''

JUDGE_USER = '''
The task is to train a robot to walk while following a desired command (comprising linear and angular velocities) and the desired gait (including the intended foot contact). 
Task note: {task_note}.

Below are the evaluation results for different reward parameters. Please identify and output the index of the best reward parameter set in the following format:
```best
index
```
Important: The index **must** be an integer which matches the index of reward parameters. 
Take care of the format of this output. Any difference is not allowed for processablity. 
You need to explain your idea. 
In locomotion task, you must much more focus on termination rate and linear velocity error, which show its stability and movability. 
'''

TRAIN_FEEDBACK = '''
We trained a reinforcement learning (RL) policy using the provided reward function code. 

## Training result
During training, we tracked the following every {epoch_freq} epochs as well as the maximum, mean, and minimum values:
    (1) Individual reward components: The values of each term in the reward function.
    (2) Global policy metrics: Episode rewards and episode lengths.

The overall training result was:.
'''

EVAL_FEEDBACK = '''
## Evaluation result

In addition to the reward terms, we computed user-guidance evaluation metrics. These metrics are directly related to some of the reward terms and are used to judge the final performance of the trained policy. All metrics (except termination) range from 0 to 1. If one metric is significantly worse than others, focus on improving the corresponding reward term in your response.
Metrics for Evaluation:
``` python
{metric_function}
```
The evaluation result is:
'''

CODE_FEEDBACK = '''
## Task
Please carefully analyze the policy feedback and provide an improved reward function to better solve the task. Use the following tips to guide your analysis:
    (1) Short episode lengths: If the episode length is consistently much shorter than the maximum ({max_episode_length}), the policy is failing to survive. In this case, you must rewrite the entire reward function to prioritize stability and survival.
    (2) If the values for a certain reward component are near identical throughout, then this means RL is not able to optimize this component as it is written. You may consider
        (a) Adjusting its scale or temperature parameter.
        (b) Rewriting the reward component.
        (c) Removing the reward component if it is unnecessary.
    (3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range
    (4) If some evaluation metric value is bad, then you must pay more attention to its corresponding reward term.
Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.
'''