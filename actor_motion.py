import glob
import math


class ActorMotion:
    def __init__(self, car_num):
        path = glob.glob('./car_info/car_%d.txt' % car_num)[0]
        fp = open(path, 'r')

        self.t = []
        self.v = []
        self.psi = []

        temp = fp.readline().split(',')
        v_temp = [float(a) for a in temp]
        self.t.append(v_temp)

        temp = fp.readline().split(',')
        v_temp = [float(a) for a in temp]
        self.v.append(v_temp)

        temp = fp.readline().split(',')
        psi_temp = [math.radians(float(a)) for a in temp]
        self.psi.append(psi_temp)

        fp.close()
        self.t = self.t[0]
        self.psi = self.psi[0]
        self.v = self.v[0]

    def get_motion_at_t(self, t):
        idx = self.t.index(t)
        return [self.v[idx], self.psi[idx]]


def main():
    pass


if __name__ == '__main__':
    main()