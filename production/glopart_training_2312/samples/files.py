#lines = ["Hello", "World", "This", "Is", "A", "Test"]
import os
path = "/publicfs/cms/user/zhaoyz/condor_output/sm/TTbar_semilep_part1_25M/"
prefix = "sm/TTbar_semilep_part1_25M/"
with open('TTbar_semilep_part1_25M.txt', 'w') as outputfile:
    for file in os.listdir(path):
        outputfile.write(prefix + file + '\n')

path = "/publicfs/cms/user/zhaoyz/condor_output/sm/TTbar_semilep_part2_25M/"
prefix = "sm/TTbar_semilep_part2_25M/"
with open('TTbar_semilep_part2_25M.txt', 'w') as outputfile:
    for file in os.listdir(path):
        outputfile.write(prefix + file + '\n')

path = "/publicfs/cms/user/zhaoyz/condor_output/sm/TTbar_semilep_part3_5M/"
prefix = "sm/TTbar_semilep_part3_5M/"
with open('TTbar_semilep_part3_5M.txt', 'w') as outputfile:
    for file in os.listdir(path):
        outputfile.write(prefix + file + '\n')

path = "/publicfs/cms/user/zhaoyz/condor_output/sm/TTbar_semilep_part4_20M/"
prefix = "sm/TTbar_semilep_part4_20M/"
with open('TTbar_semilep_part4_20M.txt', 'w') as outputfile:
    for file in os.listdir(path):
        outputfile.write(prefix + file + '\n')

path = "/publicfs/cms/user/zhaoyz/condor_output/sm/TTbar_semilep_part5_25M/"
prefix = "sm/TTbar_semilep_part5_25M/"
with open('TTbar_semilep_part5_25M.txt', 'w') as outputfile:
    for file in os.listdir(path):
        outputfile.write(prefix + file + '\n')

path = "/publicfs/cms/user/zhaoyz/condor_output/sm/WJetsToLNu_nocut_set1_25M/"
prefix = "sm/WJetsToLNu_nocut_set1_25M/"
with open('WJetsToLNu_nocut_set1_25M.txt', 'w') as outputfile:
    for file in os.listdir(path):
        outputfile.write(prefix + file + '\n')

path = "/publicfs/cms/user/zhaoyz/condor_output/sm/WJetsToLNu_nocut_set2_25M/"
prefix = "sm/WJetsToLNu_nocut_set2_25M/"
with open('WJetsToLNu_nocut_set2_25M.txt', 'w') as outputfile:
    for file in os.listdir(path):
        outputfile.write(prefix + file + '\n')
