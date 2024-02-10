.PHONY: all decompress backend frontend compress

all:
	$(MAKE) decompress
	$(MAKE) backend
	$(MAKE) frontend

decompress:
	cd food-fables-backend && gunzip -k models/veg-classifier.h5.gz

backend:
	cd food-fables-backend && . env/bin/activate && flask run &

frontend:
	cd food-fables && npm run build && npm start

compress:
	cd food-fables-backend && gzip models/veg-classifier.h5
