
import pickle

#%% mnemonics dictionary
with open('data/alias_dict.pickle', 'rb') as f:
    alias_dict = pickle.load(f)

# given a mnemonic, find all of its alias
def get_alias(mnemonic, alias_dict=None):
    alias_dict = alias_dict or {}
    return [k for k, v in alias_dict.items() if mnemonic in v]

# given a alias, find its corresponding one and only mnemonic
def get_mnemonic(alias = None, alias_dict=None):
    alias_dict = alias_dict or {}
    try:
        return [v for k, v in alias_dict.items() if alias == k][0]
    except:
        return []

if __name__ == '__main__':
    print(get_alias('DPHI', alias_dict=alias_dict))
    print(get_mnemonic(alias = 'GR', alias_dict=alias_dict))