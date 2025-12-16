tmux new-session -d -s phm
tmux send-keys -t phm "source avenv phm && jupyter lab --ip=0.0.0.0 --port=5005" C-m
tmux split-window -h
tmux send-keys -t phm "ngrok http 5005" C-m
tmux attach-session -t phm
