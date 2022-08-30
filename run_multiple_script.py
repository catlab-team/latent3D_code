import subprocess
from utils import create_result_subdir
from config import args


#--------------------------ALL TEXT LISTS-----------------------------------#
# text_lists = [
#     ['woman', 'female', 'feminine woman', 'feminine person', 'feminine human'], 
#     ['man', 'male', 'masculine man', 'masculine person', 'masculine human'],
#     ['woman with a makeup', 'human with a makeup', 'woman with a lipstick', 'beautiful woman'],

#     ['child', 'kid'], 
#     ['old human', 'old person', 'aged human', 'aged person'], 
   
#     ['large eyed human', 'human with large eyes', 'large eyes', 'big eyes', 'human with big eyes', 'person with huge eyes'], 
#     ['big lips', 'human with big lips', 'big lipped human', 'big lips', 'huge lips', 'bigger lips'], 
#     ['man with a beard', 'bearded man', 'bearded guy', 'guy with a beard'],

#     ['asian human', 'asian person'],
#     ['indian human', 'indian person'],
#     ['tanned human', 'tanned person', 'black tanned person', 'black tanned human', 'black human', 'brown tanned person'],    
    
#     ['sad human', 'sad person', 'depressed human', 'depressed person', 'melancholic', 'sad'], 
#     ['angry human', 'angry person', 'furious human', 'furious person'],
#     ['happy human', 'happy person', 'excited', 'joyful', 'laughing person'], 
#     ['afraid human', 'afraid person','horrified human', 'horrified person', 'terrified'],
#     ['disgusted human', 'disgusted person'],
    
#     ['fat human face', 'fat person face', 'fat person', 'weighted person', 'obese'], 
#     ['thin human', 'slim person', 'slim face', 'thin face']
# ]
#-----------------------------------------------------------------------------#

seeds = [1000, 81, 71]
text_lists = [
    ["makeup", ['woman with a makeup', 'human with a makeup', 'woman with a lipstick', 'beautiful woman']],
    ["largeeyes", ['large eyed human', 'human with large eyes', 'large eyes', 'big eyes', 'human with big eyes', 'person with huge eyes']],
    ["old", ['old human', 'old person', 'aged human', 'aged person']],
    ["young", ['child', 'kid', 'young human', 'young person']]
]
lambda_ids = [0.00, 0.01, 0.1, 1]

lambda_l2=0.001
mode="text-based"
num_epochs=100
learning_rate=0.01

create_result_subdir(args.result_dir, "RUN_BACKGROUND_SCRIPT_STARTED")

for seed in seeds:
    for folder_title, text_list in text_lists:
        for lambda_id in lambda_ids:
            command = f"""python3 optimize.py --seed {seed} --mode {mode} --text_list {'"' + '" "'.join(text_list) + '"'}  --num_epochs {num_epochs} --lambda_id {lambda_id} --lambda_l2 {lambda_l2} --learning_rate {learning_rate} --folder_title {folder_title}"""
            print("Running:", command)
            subprocess.run(command, shell=True)
