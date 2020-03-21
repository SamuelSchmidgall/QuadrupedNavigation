import pickle
import matplotlib.pyplot as plt


with open("/home/sam/PycharmProjects/RodentNavigation/Networks/PolicyTypes/HalfCheetah/reward_layer3_dump.pkl", 'rb') as f:
    a = pickle.load(f)

print(a)

def run_avg(l, sz=5):
    lst = list()
    avg = [sum(l[:sz])/sz for _ in range(sz)]
    for _v in l:
        avg.append(_v)
        avg.pop(0)
        lst.append(sum(avg)/sz)
    return lst

x = run_avg(a)
y = [_ for _ in range(len(a))]

plt.plot(y, x)

plt.show()











