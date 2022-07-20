from config import get_config
from network import Network
from data.data import load_data

#######################################################
def main():
    config = get_config()
    regnet3d = Network(config)
    data = load_data()
    regnet3d.test(data)

#######################################################
if __name__ == "__main__":
    main()
