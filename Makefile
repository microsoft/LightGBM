include image.env

base-image:
	docker build --no-cache -t nonsense/dask-lgb-test-base:123 - < Dockerfile-base

docker-image:
	docker build --no-cache -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile .

start-notebook:
	docker run \
		-v $$(pwd):/home/jovyan/testing \
		-p 8888:8888 \
		-p 8787:8787 \
		--name ${CONTAINER_NAME} \
		${IMAGE_NAME}:${IMAGE_TAG}

stop-notebook:
	@docker kill ${CONTAINER_NAME}
	@docker rm ${CONTAINER_NAME}
