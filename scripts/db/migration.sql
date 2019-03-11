-- liquid vision database entities

drop table IF EXISTS public.lv_usd_conv_rate;
CREATE TABLE public.lv_usd_conv_rate (
    ts timestamp NOT NULL,
    lhs_ccy text NOT NULL,
    rhs_ccy text NOT NULL,
    rate real NOT NULL
);


drop index IF exists ix_lv_usd_conv_rate;
CREATE UNIQUE INDEX ix_lv_usd_conv_rate ON public.lv_usd_conv_rate(ts, lhs_ccy, rhs_ccy);


-- only fx rates in the form USD/CCY are considered
create or replace function lv_usd_conversion_rate(
   asof timestamp,
   ccy_from text,
   ccy_to text) 
	returns real as $$ 
select
	case when ccy_to = ccy_from then 1.0::real 
		 when ccy_to = 'USD' then (1.0 / lhs.rate)::real 
		 else ((1.0 / lhs.rate) * rhs.rate)::real 
	end
from 
	((select distinct on (fx.rhs_ccy) 
			fx.rate
		from
			lv_usd_conv_rate fx
		where
			fx.lhs_ccy = 'USD'
			and fx.rhs_ccy = ccy_from
			and fx.ts <= asof
		order by
			fx.rhs_ccy,
			fx.ts desc)
	union all
		select 'Nan'
	) lhs,
	((select
			distinct on
			(fx.rhs_ccy) fx.rate
		from
			lv_usd_conv_rate fx
		where
			fx.lhs_ccy = 'USD'
			and fx.rhs_ccy = ccy_to
			and fx.ts <= asof
		order by
			fx.rhs_ccy,
			fx.ts desc)
	union all
		select 'Nan' 
	) rhs
limit 1 
$$ language sql;


drop table if exists public.lv_trader_perf cascade;
CREATE TABLE public.lv_trader_perf (
    id bigserial PRIMARY KEY,
	epoch int4 NOT NULL,
	duration int4 NOT null,
	user_id text NULL,
	-- null product_id represents portfolio
	product_id int4 NULL,
	-- product pricing, for TCA analysis
	px_open real NULL,
	px_high real NULL,
	px_low real NULL,
	px_close real NULL,
	product_cb real NULL,
	-- position ccy
	pos_ccy text NULL,
	nop_open real null,
	nop_high real null,
	nop_low real null,
	nop_close real null,
	-- pnl attributed to product (or whole portfolio if product_id is null)
	trade_cnt int4 not null default 0,
	win int4 not null default 0,
	loss int4 not null default 0,	
	pnl_ccy text not null,
	pnl_open real NOT null default 0,
	pnl_high real NOT null default 0,
	pnl_low real NOT null default 0,
	pnl_close real NOT null default 0
);

drop view if exists public.v_lv_trader_perf;
CREATE VIEW public.v_lv_trader_perf AS select
	epoch,
	0::int4 whence,
	duration,
	product_id,
	user_id,
	pos_ccy,
	nop_open,
	nop_high,
	nop_low,
	nop_close,
	-- hide pricing for now
	trade_cnt,
	win,
	loss,
	pnl_ccy,
	pnl_open,
	pnl_high,
	pnl_low,
	pnl_close,
	0::real as roi
from 
	public.lv_trader_perf
order by
	epoch asc, product_id
;


drop index if exists ix_lv_trader_perf;
CREATE INDEX ix_lv_trader_perf ON public.lv_trader_perf(user_id, epoch, product_id);

drop domain report_limit;
create domain report_limit as integer
-- reject requests  
check (value >= 0 and value <= 367*24);

create or replace function lv_date_boundary(userid text, start_dt timestamp, period interval) returns timestamp with time zone as $$
	select case when (period < interval '0') then least(start_dt, to_timestamp(max(epoch))) 
				else greatest(start_dt, to_timestamp(min(epoch))) 
			end 	
	from v_lv_trader_perf 
	where user_id = userid
$$ language sql;

		
create or replace
	function lv_trader_perf_over_time(userid text,
	from_dt timestamp,
	to_dt timestamp,
	period text,
	eval_ccy text) 
 	returns setof public.v_lv_trader_perf as $$ 
  with x as (
  select
		case when (period::interval < interval '0') then extract('epoch' from t1 + period::interval) else extract('epoch' from t1) end as t1,
		case when (period::interval < interval '0') then extract('epoch' from t1) else extract('epoch' from t1 + period::interval) end as t2
  from
		generate_series(date_trunc('minute', lv_date_boundary(userid, from_dt, period::interval)),
		to_dt,
		period::interval) as t1
	-- arbitrary clamp
	limit greatest( abs((extract('epoch' from (to_dt - from_dt))::int/extract('epoch' from period::interval)::int))::report_limit, 367*24)
  ) select
  		t,
		(y_first).epoch,
		greatest(d, (y_last).epoch + (y_last).duration) - (y_first).epoch,
		product_id::int4,
		userid as user_id,
		nop_ccy,
		(y_first).nop_open,
		nop_max,
		nop_min,
		(y_last).nop_close,
		trade_cnt::int4,
		win::int4,
		loss::int4,
		eval_ccy,
		rate * (y_first).pnl_open,
		rate * pnl_max,
		rate * pnl_min,
		rate * (y_last).pnl_close,
		0.0::real
	from
		(
		select distinct on (1, 2) 
			x.t1::int4 as t,
			y.product_id as product_id,
			x.t2::int4 as d,
			first_value(y) over w as y_first,
			last_value(y) over w as y_last,
			sum(y.trade_cnt) over w as trade_cnt,
			sum(y.win) over w as win,
			sum(y.loss) over w as loss,
			pos_ccy as nop_ccy ,
			last_value (case when y.pnl_ccy = eval_ccy then 1.0 else lv_usd_conversion_rate(to_timestamp(y.epoch)::timestamp without time zone, y.pnl_ccy, eval_ccy) end) over w as rate ,
			max(nop_high) over w as nop_max ,
			min(nop_low) over w as nop_min ,
			max(pnl_high) over w as pnl_max ,
			min(pnl_low) over w as pnl_min 
		from
			x
		join public.v_lv_trader_perf y on
			y.epoch >= x.t1
			and y.epoch < x.t2
		where
			y.user_id = userid
		window w as (
			partition by x.t1, y.product_id 
			order by y.epoch asc, y.product_id
			ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED following
		) 
  ) ohlc

$$ language sql;


drop table if exists public.lv_trader_balance cascade;
CREATE TABLE public.lv_trader_balance (
    id bigserial PRIMARY KEY,
	user_id text NOT NULL,
	epoch int4 NOT NULL,
	asset text NULL,
	units real null default 0
);

drop view if exists public.v_lv_trader_balance;
CREATE VIEW public.v_lv_trader_balance AS SELECT 
	epoch,
	0::int4 whence,
	0::int4 duration,
	asset,
	1 as xfer_cnt,
	units as units_open,
	units as units_high,
	units as units_low,
	units as units_close,
	null::text as ccy,
	null::real as fx_rate,
	user_id
from 
	public.lv_trader_balance
order by
	epoch,
	asset 
;

drop index if exists ix_lv_trader_balance;
CREATE INDEX ix_lv_trader_balance ON public.lv_trader_balance(user_id, epoch);

-- query balance
create or replace
	function lv_trader_balance_over_time(userid text,
	from_dt timestamp,
	to_dt timestamp,
	period text,
	eval_ccy text) returns setof public.v_lv_trader_balance as $$ with x as (
	select
		case when (period::interval < interval '0') then extract('epoch' from t1 + period::interval) else extract('epoch' from t1) end as t1,
		case when (period::interval < interval '0') then extract('epoch' from t1) else extract('epoch' from t1 + period::interval) end as t2
	from
		generate_series(date_trunc('minute', lv_date_boundary(userid, from_dt, period::interval)),
		to_dt,
		period::interval) as t1
	limit 24 * 60 * 31
	-- arbitrary clamp
) select
		t,
		(y_first).epoch as epoch,
		greatest((y_last).epoch - (y_first).epoch, 1) as duration,
		asset,
		xfer_cnt::int4,
		(y_first).units_open,
		units_max,
		units_min,
		(y_last).units_close,
		eval_ccy,
		rate,
		userid as user_id
	from
		(
		select distinct on (1, 2) 
			x.t1::int4 as t ,
			y.asset as asset ,
			x.t2::int4 as d ,
			first_value(y) over w as y_first ,
			last_value(y) over w as y_last ,
			sum(xfer_cnt) over w as xfer_cnt ,
			first_value(units_open) over w as open ,
			max(units_high) over w as units_max ,
			min(units_low) over w as units_min ,
			last_value(units_close) over w as close,
			last_value (case when y.asset = eval_ccy then 1.0 else lv_usd_conversion_rate(to_timestamp(y.epoch)::timestamp without time zone, y.asset, eval_ccy) end) over w as rate
		from
			x
		join public.v_lv_trader_balance y on
			y.epoch >= x.t1
			and y.epoch < x.t2
		where
			y.user_id = userid 
		window w as (
			partition by x.t1, y.asset 
			order by y.epoch asc
			ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED following
		) 
	) ohlc
$$ language sql;

