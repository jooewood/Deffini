all:
	docker login docker.yfish.x
	docker build --tag fernie:v2.0.0 .
	docker tag fernie:v2.0.0 fernie:latest

