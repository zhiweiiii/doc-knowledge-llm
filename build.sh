git pull
echo y |docker image prune
docker compose -f 'docker-compose.yml' up -d --build kn
docker compose logs -f --tail 100 kn