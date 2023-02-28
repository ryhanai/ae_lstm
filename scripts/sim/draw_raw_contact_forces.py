import numpy as np

bin_state = test_data[37][2]

def draw_raw_contact_forces():
    cps = s.p.getContactPoints()
    viewer.publish_bin_state(bin_state, fmap)
    for cp in cps:
        if cp[1] != 0 and cp[9] > 1e-5:
            b1 = cp[3]
            b2 = cp[4]
            pos = np.array(cp[6])
            direction = np.array(cp[7])
            if direction[2] < 0:
                direction = -direction
            fval = cp[9]
            print(cp[9])
            viewer.rviz_client.draw_arrow(pos, pos + direction * fval * 0.1 , [1,0,0,1], [0.005, 0.02, 0.01])
    viewer.rviz_client.show()
