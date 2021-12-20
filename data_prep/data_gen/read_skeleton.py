import numpy as np


def read_skeleton(file):
    matx = np.loadtxt(file, delimiter=',')
    skeleton_sequence = {}
    skeleton_sequence['numFrame'] = matx.shape[0]
    skeleton_sequence['frameInfo'] = []
    for t in range(skeleton_sequence['numFrame']):
        frame_info = {}
        frame_info['numBody'] = 1
        frame_info['bodyInfo'] = []
        for m in range(frame_info['numBody']):
            body_info = {}
            body_info['numJoint'] = matx.shape[1]
            body_info['jointInfo'] = []
            iterator = [0, 3, 6, 9, 12, 15, 18, 21, 24]
            for v in iterator:
                joint_info_key = [
                    'x', 'y', 'z'
                ]
                joint_info = {
                    k: float(v)
                    for k, v in zip(joint_info_key, matx[t,[v,v+1,v+2]])
                }
                body_info['jointInfo'].append(joint_info)
            frame_info['bodyInfo'].append(body_info)
        skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=1, num_joint=9):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data
