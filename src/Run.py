import src.Train as Train

def main():
    Train.trainGAN(epochs=50, checkpoint=True)

if __name__ == "__main__":
    main()