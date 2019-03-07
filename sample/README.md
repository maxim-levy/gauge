Here is an example of how you could load a CSV file:

`COPY public.lv_usd_conv_rate (ts, lhs_ccy, rhs_ccy, rate) FROM '/Users/markhammond/src/quoine/gauge/sample/mock_fx_rate.csv' DELIMITER E',' CSV HEADER;`
