import sys
import random

# user_arm_tag = []

# #remove duplicate events
# fin = open('raw_data/ratings.csv', 'r')
# fin.readline()
# last = {}
# for line in fin:
#     arr = line.strip().split(',')
#     if float(arr[2]) < 1:
#         continue
#     t = {}
#     t['uid'] = int(arr[0])
#     t['aid'] = int(arr[1])
#     t['tstamp'] = int(arr[3])
#     #print t['tstamp']
#     if not t == last:
#         last = t
#         user_arm_tag.append(t)
# print('event number: '+str(len(user_arm_tag)))

# #filter arm pool for each user
# user_arm_pool = {}
# arm_pool = set()
# for t in user_arm_tag:
#     arm_pool.add(t['aid'])

# for t in user_arm_tag:
#     if not (t['uid'] in user_arm_pool):
#         user_arm_pool[t['uid']] = arm_pool.copy()
#     if t['aid'] in user_arm_pool[t['uid']]:
#         user_arm_pool[t['uid']].remove(t['aid'])
# # random.shuffle(user_arm_tag)

# #generate random arm_pool and write to file
# fout = open('/processed_data/processed_events_orderedByUserTime.dat','w')
# fout.write('userid  timestamp   arm_pool\n')
# for t in user_arm_tag:
#     random_pool = [t['aid']]+random.sample(user_arm_pool[t['uid']], 24)
#     fout.write(str(t['uid'])+'\t'+str(t['tstamp'])+'\t'+str(random_pool)+'\n')
# fout.close()

#remove duplicate events

threshold = 3000
num_users = 0
arm_pool = set()
user2ItemSeqs = {}
temp_user_arm_tag = []
cur_uid = 1
fin = open('raw_data/ratings.csv', 'r')
fin.readline()
last = {}
for line in fin:
    arr = line.strip().split(',')
    if float(arr[2]) < 1:
        continue
    t = {}
    t['uid'] = int(arr[0])
    t['aid'] = int(arr[1])
    t['tstamp'] = int(arr[3])
    arm_pool.add(t['aid'])
    if cur_uid == t['uid']:
        temp_user_arm_tag.append(t)
    else:
        if len(temp_user_arm_tag) > threshold:
            num_users += 1
            user2ItemSeqs[cur_uid] = temp_user_arm_tag
        cur_uid = t['uid']
        temp_user_arm_tag = []
        temp_user_arm_tag.append(t)
print('user number: '+str(len(user2ItemSeqs)))
print('item number: '+str(len(arm_pool)))

#filter arm pool for each user
user_arm_pool = {}

for uid,ItemSeqs in user2ItemSeqs.items():
    for t in ItemSeqs:
        if not (t['uid'] in user_arm_pool):
            user_arm_pool[t['uid']] = arm_pool.copy()
        if t['aid'] in user_arm_pool[t['uid']]:
            user_arm_pool[t['uid']].remove(t['aid'])
# random.shuffle(user_arm_tag)

file = open("./processed_data/randUserOrderedTime_N{}_ObsMoreThan{}.dat".format(len(user2ItemSeqs), threshold),"w")
file.write('userid  timestamp   arm_pool\n')
global_time = 0
while user2ItemSeqs:
    userID = random.choice(list(user2ItemSeqs.keys()))
    t = user2ItemSeqs[userID].pop(0)
    global_time += 1
    random_pool = [t['aid']]+random.sample(user_arm_pool[t['uid']], 24)
    file.write(str(t['uid'])+'\t'+str(t['tstamp'])+'\t'+str(random_pool)+'\n')
    if not user2ItemSeqs[userID]:
        del user2ItemSeqs[userID]
file.close()
print("global_time {}".format(global_time))
# #generate random arm_pool and write to file
# fout = open('/processed_data/processed_events_orderedByUserTime.dat','w')
# fout.write('userid  timestamp   arm_pool\n')
# for t in user_arm_tag:
#     random_pool = [t['aid']]+random.sample(user_arm_pool[t['uid']], 24)
#     fout.write(str(t['uid'])+'\t'+str(t['tstamp'])+'\t'+str(random_pool)+'\n')
# fout.close()