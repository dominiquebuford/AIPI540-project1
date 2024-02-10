#!/usr/bin/python3

from data.s3bucket_googlecolab.py import setup_and_download_from_s3
from models.main_inception_vegtable.py import train_and_finetune_model

def main(local_dir=os.getcwd()):
    setup_and_download_from_s3('AKIA6GBMB6KOCGR7U6EQ', 'JibunchrhyB26LoMtnT1ZvjSS5byH3XtJUwVEGiW', 'vegetablepictures', 'train', os.path.join(local_dir, 'train'))

    setup_and_download_from_s3('AKIA6GBMB6KOCGR7U6EQ', 'JibunchrhyB26LoMtnT1ZvjSS5byH3XtJUwVEGiW', 'vegetablepictures', 'validation', os.path.join(local_dir, 'validation'))

    train_and_finetune_model(os.path.join(local_dir, 'train'), os.path.join(local_dir, 'validation'))

if __name__ == "__main__":
    main()
