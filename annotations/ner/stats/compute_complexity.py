# file = 'manual-ann-ner-extended.yaml'
# file2 = 'manual-ann-ner.yaml'

import yaml

# with open(file, 'r') as f:
#     anns = yaml.safe_load(f)
# with open(file2, 'r') as f:
#     anns2 = yaml.safe_load(f)



with open("../data/manual-ann-ner-fp.yaml", 'r') as f:
    anns = yaml.safe_load(f)

anns_tot = []
for ann_S in [anns]: #, anns2]:
    ann_S = ann_S['annotations']
    for ann in ann_S.values():
        anns_tot.append(ann['structured'])

# print(anns_tot)
# print(len(anns_tot))

mo = 0
nb_keys = 0
object_alone = 0
object_and_measure = 0
object_and_time = 0
object_and_measure_and_specifier = 0
object_and_measure_and_time = 0
composite = 0

for anns in anns_tot:
    for ann in anns:
        if "object" in ann.keys():
            mo += 1
        nb_keys += len(ann)
    if len(anns) == 1:
        if len(anns[0].keys()) == 1 and 'object' in anns[0].keys():
            object_alone += 1
        if len(anns[0].keys()) == 2 \
             and 'object' in anns[0].keys() \
             and 'measure' in anns[0].keys():
            object_and_measure += 1
        if len(anns[0].keys()) == 2 \
             and 'object' in anns[0].keys() \
             and 'time' in anns[0].keys():
            object_and_time += 1
        if len(anns[0].keys()) == 3 \
             and 'object' in anns[0].keys() \
             and 'measure' in anns[0].keys() \
             and 'specifier' in anns[0].keys():
            object_and_measure_and_specifier += 1
        if len(anns[0].keys()) == 3 \
             and 'object' in anns[0].keys() \
             and 'measure' in anns[0].keys() \
             and 'time' in anns[0].keys():
            object_and_measure_and_time += 1
    if len(anns) > 1:
        composite += 1

print(mo)
print(nb_keys/len(anns_tot))
# print(object_alone, object_alone/len(anns_tot))
# print(object_and_measure)
# print(object_and_measure_and_specifier)
# print(composite)
# print(object_and_time)
# print(object_and_measure_and_time)

# Assuming anns_tot is the total count
total_count = len(anns_tot)

# Normalize values
object_alone_normalized = object_alone / total_count
object_and_measure_normalized = object_and_measure / total_count
object_and_measure_and_specifier_normalized = object_and_measure_and_specifier / total_count
composite_normalized = composite / total_count
object_and_time_normalized = object_and_time / total_count
object_and_measure_and_time_normalized = object_and_measure_and_time / total_count

# Print normalized values
print(object_alone, object_alone_normalized)
print(object_and_measure, object_and_measure_normalized)
print(object_and_measure_and_specifier, object_and_measure_and_specifier_normalized)
print(composite, composite_normalized)
print(object_and_time, object_and_time_normalized)
print(object_and_measure_and_time, object_and_measure_and_time_normalized)