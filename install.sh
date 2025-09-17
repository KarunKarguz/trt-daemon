sudo cp build/trt-daemon /usr/local/bin/
sudo mkdir -p /opt/model && sudo cp model/model_fp32.plan /opt/model/
sudo cp deploy/trt-daemon.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trt-daemon
sudo systemctl status trt-daemon
sudo journalctl -u trt-daemon -f