docker run --name pa_scs pa_scs -v common/data/training/TIMIT:/usr/local/common/data/training/TIMIT
docker exec -t -i pa_scs /bin/bash
pause