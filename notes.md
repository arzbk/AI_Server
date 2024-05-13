Requirements generated with "pipreqs ." in root dir of project

# IF SERVER PROCESS GET'S STUCK: RUN...
1. sudo ss -lptn 'sport = :3000'
2. sudo kill -9 <PID> (kill bentoml proc first, then python proc, or it'll keep respawning)

# Alias for doing this efficiently
alias kill3000="fuser -k -n tcp 3000" && kill3000

# For running non-default services...
COQUI_TOS_AGREED=1 bentoml serve test:TTSService <--- NOTE the licensing agreement