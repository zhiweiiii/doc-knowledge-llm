git pull
echo y |docker image prune
docker-compose down --remove-orphans
docker-compose -f 'docker-compose-gpu.yml' up -d --build kn
docker-compose logs -f --tail 100 kn