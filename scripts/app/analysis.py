import pandas as pd
import numpy as np

def read_result_files(scene_numbers=range(0,100)):
    data = []
    for scene_number in scene_numbers:
        data.extend(pd.read_pickle('picking_results/{}.pkl'.format(scene_number)))
    return data


def average_for_all_liftings(data):
    print(np.average([d['metrics']['fmap']['max linear vel'] for d in data]))
    print(np.average([d['metrics']['up']['max linear vel'] for d in data]))
    print(np.average([d['metrics']['fmap']['max linear disp'] for d in data]))
    print(np.average([d['metrics']['up']['max linear disp'] for d in data]))
    print(np.average([d['metrics']['fmap']['max angular vel'] for d in data]))
    print(np.average([d['metrics']['up']['max angular vel'] for d in data]))
    print(np.average([d['metrics']['fmap']['max augular disp'] for d in data]))
    print(np.average([d['metrics']['up']['max augular disp'] for d in data]))


def filtering(data, z_threshold=0.5):
    filtered_data = []
    for d in data:
        lifting_vel = d['pick screw'][0]
        lifting_direction = lifting_vel / np.linalg.norm(lifting_vel)
        if lifting_direction[2] < z_threshold:
            filtered_data.append(d)
    return filtered_data


def extract_trials_with_no_initial_collision(data):
    filtered_data = []
    for d in data:
        ic = d['initial collisions']
        skip_this = False
        for cp in ic:
            body1, body2 = sorted((cp[1], cp[2]))
            target_body = s.object_cache[d['target']].get_body()
            if target_body == body2 and body1 == 2 and cp[5][2] > 0.74:
                skip_this = True
                break
        if not skip_this:
            filtered_data.append(d)
    return filtered_data
        
# all 499 trials
# 0.3078399707496754
# 0.3009437992903435
# 0.030620512213035706
# 0.03397111813465767
# 37.19793477485451
# 31.091641604153793
# 0.3004981453593149
# 0.2759540512629854

# lifting less than 30 degree, 52 trials
# 0.43208783331765704
# 0.4777267115958742
# 0.04150407512633307
# 0.05625379409087909
# 60.97176731611902
# 118.97404054938649
# 0.42959435923047007
# 0.7103927579876863

# lifting objects with no initial collision with wall
