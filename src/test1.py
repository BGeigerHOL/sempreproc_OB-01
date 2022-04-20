import json

# Opening JSON file
f = open('config.json')

# returns JSON object as
# a dictionary
config = json.load(f)

# Iterating through the json
# list
maskdot = config['maskdot'][0]
for k,v in maskdot.items():
    print(k,v)

# Closing file
f.close()