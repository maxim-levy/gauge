#!/bin/sh

mkdir -p ../output && \
curl 'https://api.liquid.com/products/5/ohlc?resolution=60' -H 'origin: https://chart.liquid.com' -H 'accept-encoding: gzip, deflate, br' -H 'accept-language: en-US,en;q=0.9' -H 'authority: api.liquid.com' -H 'x-quoine-preferred-language: en' -H 'x-quoine-vendor-id: 3' -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36' -H 'x-quoine-client-version: 2' -H 'x-quoine-device: web-native' -H 'accept: application/json' -H 'referer: https://chart.liquid.com/?provider=trading-view&guest=1&symbol=BTCJPY&productId=5&language=en&theme=blue&precision=2' -H 'x-quoine-api-version: 2' -H 'if-none-match: "1bcd1f0395d19602f3d0b5892a169673"' -H 'content-type: application/json' --compressed \
	| jq -r '["Date","Open","High","Low","Close","Volume"],.data[] | @csv' \
	> ../output/btc_jpy.json && echo "wrote file ../output/btc_jpy.json" > /dev/tty

