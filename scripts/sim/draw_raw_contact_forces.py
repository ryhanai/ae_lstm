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
            viewer.rviz_client.draw_arrow(pos, pos + direction * fval * 0.1, [1, 0, 0, 1], [0.005, 0.02, 0.01])
    viewer.rviz_client.show()


def draw_isaac_raw_contact_state(scale=5.0):
    cs = pd.read_pickle("/home/ryo/Program/moonshot/ae_lstm/scripts/sim/data/contact_raw_data00000.pkl")
    bs = pd.read_pickle("/home/ryo/Program/moonshot/ae_lstm/scripts/sim/data/bin_state00000.pkl")
    viewer.publish_bin_state(bs, fmap)
    for p, f, n in zip(cs[0], cs[1], cs[3]):
        l = f * scale
        l = min(l, 0.2)
        if n[2] < 0:
            n = -n
        viewer.rviz_client.draw_arrow(p, p + l * n, [1, 0, 0, 1], [0.005, 0.02, 0.01])
    viewer.rviz_client.show()
