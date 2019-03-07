#!/usr/bin/env python3
# -*- coding: utf-8; py-indent-offset:4 -*-
"""Simulate trading performance with reference to external trade history using walk-backward strategy
https://en.wikipedia.org/wiki/Walk_forward_optimization

Starting with a recent snaphot of account value and open positions, we walk backwards in time, 
replaying market events to reconstruct the trader's pnl profile. In the walk backwards run we
focus on collecting product level data, rather than portfolio statistics.

wrote report <_io.TextIOWrapper name='output/report.out.csv' mode='w' encoding='UTF-8'>
wrote report <_io.TextIOWrapper name='output/balance.out.csv' mode='w' encoding='UTF-8'>
Backward and forward walk within tolerance! (0.001%)

FIXME fixup todate and fromdate handling to limit how far we go back in history (maybe rely min distance, limited by CPU time)
FIXME correct mark-to-market (mtm) by using price by size
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from datetime import datetime
from decimal import Decimal
from more_itertools import peekable

import bisect
import datetime
import decimal
import argparse
import calendar
import logging
import csv
import json
import os
import math
import pandas as pd
import sys

__author__ = "Mark Hammond"
__copyright__ = "Copyright 2019, Quoine Financial"
__credits__ = ["Mark Hammond"]
__license__ = "None"
__version__ = "0.0.1"
__maintainer__ = "Mark Hammond"
__email__ = "mark@quoine.com"
__status__ = "Development"

import bt
from bt import walkforward

logging.basicConfig(level=logging.WARN)

sign = lambda x: math.copysign(1, x)

def round_sigfig(x, sig=5):
    if x == 0: 
        return 0 
    else:
        return round(x, sig-int(math.floor(math.log10(abs(x))))-1)     

def pgtime_to_epoch(datestring):
    """
    pgtime_to_epoch - convert the iso8601 date into the unix epoch time
    >>> pgtime_to_epoch("2012-07-09 22:27:50")
    1341872870
    """
    return calendar.timegm(datetime.datetime.strptime(datestring, '%Y-%m-%d %H:%M:%S').timetuple())

def iso8601_to_epoch(datestring):
    """
    iso8601_to_epoch - convert the iso8601 date into the unix epoch time
    >>> iso8601_to_epoch("2012-07-09T22:27:50")
    1341872870
    """
    return calendar.timegm(datetime.datetime.strptime(datestring, "%Y-%m-%dT%H:%M:%S").timetuple())


class DecimalEncoderWithPossiblePrecisionLoss(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super(DecimalEncoderWithPossiblePrecisionLoss, self).default(o)


def walkbackward(args=None):
    args = parse_args(args)

    product_static = dict()
    with open(args.product_static_json, encoding='utf-8') as data_file:
        product_static = json.loads(data_file.read())

    # prices (from influxdb)
    # ... we'll sample ladders and use ~10 levels of depth to calculate mark-to-market
    prices = pd.read_csv(
        args.data0, 
        converters={'Date': int, 'Open': decimal.Decimal},
        usecols = ['Date', 'Open', 'Product'],
     )
    prices = prices.set_index('Date')
    prices = prices[::-1]
 
    # trades (from pg_executions)
    trades = pd.read_csv(
        args.data1, 
        converters={'datetime': int, 'price': decimal.Decimal, 'quantity': decimal.Decimal},
        usecols = ['datetime', 'quantity', 'price', 'product'],
     )
    trades = trades.set_index('datetime')
    trades = trades[::-1]
       
    # prices (from pg_balance, but need historical data, not just most recent)
    funding = pd.read_csv(
        args.data2, 
        converters={'value': decimal.Decimal, 'created_at': pgtime_to_epoch},
        usecols = ['created_at', 'currency', 'value'],
    )

    funding['created_at'] = pd.to_datetime(funding['created_at'], format='%Y-%m-%d %H:%M:%S').astype(int)
    funding = funding.set_index('created_at')
    funding = funding[::-1]

    asset = dict()

    dit = peekable(prices.iterrows())
    oit = peekable(trades.iterrows())
    fit = peekable(funding.iterrows())
   
    if args.fromdate:
        fromtimestamp = iso8601_to_epoch(args.fromdate)
        prices = prices[prices.index.searchsorted(fromtimestamp):]
        trades = trades[trades.index.searchsorted(fromtimestamp):]
        funding = funding[funding.index.searchsorted(fromtimestamp):]
    
    end = [-sys.maxsize]

    data = []
    balance_data = []

    last_dt = 0
    px_dt = dit.peek(end)[0]
    od_dt = oit.peek(end)[0]
    fd_dt = fit.peek(end)[0]

    product_perf = dict()

    while od_dt != end[0] or fd_dt != end[0]:
        
        if fd_dt >= px_dt and fd_dt >= od_dt: # process cashflow
    
            fd = next(fit)[1]
            last_dt = fd_dt

            ccy = fd[0]
            qty = Decimal(fd[1])

            if ccy not in asset:
                asset[ccy] = Decimal(0)
            asset[ccy] = asset[ccy] + qty
            
            balance_data.append([last_dt, ccy, asset[ccy]])

            if fd[0] == Decimal(0):
                asset.pop(ccy)
            
            fd_dt = fit.peek(end)[0]
 
        elif od_dt >= px_dt: # process trade

            last_dt = od_dt

            noda = next(oit)[1]
            od_dt = oit.peek(end)[0] 
            
            oda_copy = noda.copy()
            oda_copy[0] = -oda_copy[0]
            pos_delta = oda_copy[0]

            pos_delta = oda_copy[0]
            assert(pos_delta != Decimal(0))

            product_id = oda_copy[2]
            assert (product_id)

            if product_id in product_perf:                
                p_pnl = product_perf[product_id]
            else:
                product_perf[product_id] = bt.pnl_tracker()
                p_pnl = product_perf[product_id]

            trade_in_same_direction = sign(p_pnl.cum_units + pos_delta) == sign(p_pnl.cum_units)

            if trade_in_same_direction:
                bt.process_order(p_pnl, oda_copy)
            else:

                oda_copy = noda.copy()

                # TODO consider reflecting closed position as cashflow
                units_closed = -p_pnl.cum_units
                units_left = pos_delta + p_pnl.cum_units

                if units_closed != 0:
                    oda_copy[0] = units_closed
                    bt.process_order(p_pnl, oda_copy, False)

                if units_left != 0:
                    oda_copy[0] = units_left
                    bt.process_order(p_pnl, oda_copy)

            cum_pnl = p_pnl.cum_pnl
 
            if logging.DEBUG >= logging.getLogger().getEffectiveLevel(): 
                print("oda {} px {} adj_pnl {:f} cum_pnl {} cum_units {:f} real {} potential {} prev_units {:f} cost_of_transaction {} cum_cost {} transacted_value {} pos_delta {:f}".format(
                    last_dt, 
                    round_sigfig(p_pnl.mtm,6),
                    round_sigfig(cum_pnl / abs(p_pnl.cum_units)) if p_pnl.cum_units != 0 else 0,
                    round_sigfig(cum_pnl, 3),
                    round_sigfig(p_pnl.cum_units,3),
                    p_pnl.realised_pnl,
                    p_pnl.potential_pnl,
                    round_sigfig(p_pnl.prev_units,3), 
                    round_sigfig(p_pnl.cost_of_transaction), 
                    round_sigfig(p_pnl.cum_cost), 
                    round_sigfig(p_pnl.transacted_value),
                    round_sigfig(pos_delta,3),
                    )
            )

        else:
 
            last_dt = px_dt
            md = next(dit)[1]
            px_dt = dit.peek(end)[0]

            product_id = md[1]  
            assert (product_id)

            if product_id in product_perf:
       
                p_pnl = product_perf[product_id]
                if p_pnl.cum_units != 0:
                    #FIXME improve accuracy by looking up price by size market data feed
                    #methodology must be consistent with walk forward
                    mtm = md[0]
                    p_pnl.mtm = mtm
                
                    p_pnl.potential_pnl = (p_pnl.mtm - p_pnl.cost_basis) * p_pnl.cum_units
                    p_pnl.cum_pnl = p_pnl.realised_pnl + p_pnl.potential_pnl             

                    if logging.DEBUG >= logging.root.level:                                                                                                                                                                                                                                                                             
                        print("mkt {} px {} adj_pnl {} cum_pnl {} cum_units {:f} prev_units {} cost_of_transaction {} cum_cost {} transacted_value {}".format(
                            last_dt, 
                            round_sigfig(p_pnl.mtm, 6), 
                            round_sigfig(p_pnl.potential_pnl, 3),
                            round_sigfig(p_pnl.cum_pnl, 3),
                            round_sigfig(p_pnl.cum_units,3), 
                            round_sigfig(p_pnl.prev_units), 
                            round_sigfig(p_pnl.cost_of_transaction), 
                            round_sigfig(p_pnl.cum_cost), 
                            round_sigfig(p_pnl.transacted_value),
                            )
                        )        
                        

        next_dt = max(px_dt, od_dt, fd_dt)

        if last_dt > next_dt: # close of timepoint

            if logging.DEBUG >= logging.getLogger().getEffectiveLevel(): 
                print("asset {}".format(asset))
            
            asset_copy = asset.copy()

            balance_eq = Decimal(0)

            for product_id, pnl in product_perf.items():
                key = str(product_id)
                pnl_ccy = product_static[key]["quoted"]
                asset_copy[pnl_ccy] += pnl.realised_pnl


            data.append([last_dt, balance_eq, asset_copy])
 
    if args.todate:
        totimestamp = iso8601_to_epoch(args.todate)
        keys = [r[0] for r in data] 
        i = bisect.bisect_left(keys, totimestamp)
        data = data[:i]      

    opening_date = data[-1][0]
    closing_balance = data[0][-1]
    print("Closing balance {}".format(closing_balance))

    implied_opening_balance = data[-1][-1]
    print("Implied opening balance {}".format(implied_opening_balance))

    implied_opening_balance = json.dumps(implied_opening_balance, cls=DecimalEncoderWithPossiblePrecisionLoss)

    new_args = [] + sys.argv[1:] \
        + ["--opening_balance",implied_opening_balance] \
        + ["--fromdate",datetime.datetime.utcfromtimestamp(opening_date).isoformat()]

    fw_balance_n_product = walkforward(new_args)
    
    simulated_closing_balance = fw_balance_n_product[0]
    simulated_product_perf = fw_balance_n_product[1]
    
    for product_id, pnl in simulated_product_perf.items():
        key = str(product_id)
        pnl_ccy = product_static[key]["quoted"]
        simulated_closing_balance[pnl_ccy] += pnl.realised_pnl

    threshold = 0.00001
    issame = sorted(simulated_closing_balance.keys()) == sorted(closing_balance.keys())
    for ccy in simulated_closing_balance.keys():
        issame = issame and math.isclose(simulated_closing_balance[ccy], closing_balance[ccy], rel_tol = threshold)

    if not issame:
        print("Backward ({}) and forward walk ({}) substantially different :-(".format(closing_balance, simulated_closing_balance))
    else:
        print("Backward and forward walk within tolerance! ({}%)".format(threshold* 100))

    return issame

    # perf = df['Returns'].calc_stats() 
    # perf.display() 
    # print(perf)   
    # perf.plot()  
            


def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            'Walk-backward historical simulation'
        )
    )

    parser.add_argument('--data0', default='sample/btc_jpy.in.csv',
                        required=False, help='Data to read in')

    # select 
    #     * 
    # from 
    #     import_from_quoine_market.pg_executions 
    # where 1 = 1
    #     and product_id = 5 
    #     and (buy_trader_id = 401130 or sell_trader_id = 401130)
    #     and (buy_trader_id <> sell_trader_id) 
    #     and created_at > (select max(updated_at) from import_from_quoine_main.pg_balances where user_id = 401130)
    # order by id asc;
    parser.add_argument('--data1', default='sample/trades.in.csv',
                        required=False, help='Data to read in')

    # select 
    #     *
    # from 
    #     import_from_quoine_main.pg_balances 
    # where 
    #     user_id = 401130    
    parser.add_argument('--data2', default='sample/pg_balances.rev.in.csv',
                        required=False, help='Data to read in')

    # Defaults for dates
    parser.add_argument('--fromdate', required=False, default='',
                        help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')

    parser.add_argument('--todate', required=False, default='',
                        help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')

    parser.add_argument('--plot', required=False, default='',
                        nargs='?', const='{}',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--product_static_json', default='sample/product.in.json', required=False,
                        help=('Json dict of product static data'))

    # parser.add_argument('--timeframe', default='minutes', required=False,
    #                     choices=['ticks', 'microseconds', 'seconds',
    #                              'minutes', 'daily', 'weekly', 'monthly'],
    #                     help='Timeframe to resample to')

    # parser.add_argument('--compression', default=1, required=False, type=int,
    #                     help=('Compress n bars into 1'))

    parser.add_argument('--writer', required=False, action='store_true',
                        help=('Add a Writer'))

    parser.add_argument('--report_csv', default='output/report.out.csv', required=False,
                        help=('Output perform data to csv'))

    parser.add_argument('--balance_csv', default='output/balance.out.csv', required=False,
                        help=('Output balance data to csv'))

    return parser.parse_args(pargs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)

    if not walkbackward():
        sys.exit(os.EX_SOFTWARE)



