#!/usr/bin/env python3
# -*- coding: utf-8; py-indent-offset:4 -*-
"""Simulate trading performance with reference to external trade history. 

Starting from a historical snapshot of account value and open positions, we walk forward in time and replay market events to reconstruct the trader's pnl profile.
https://en.wikipedia.org/wiki/Walk_forward_optimization
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from datetime import datetime
from decimal import Decimal
from more_itertools import peekable

import datetime
import decimal
import argparse
import calendar
import logging
import csv
import io
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

class pnl_tracker:
    def __init__(   self):
        self.potential_pnl = Decimal(0.0)
        self.cum_cost = Decimal(0.0)
        self.cost_basis = Decimal(0.0)
        self.realised_pnl = Decimal(0.0)
        self.prev_units = Decimal(0.0)
        self.cum_units = Decimal(0.0)
        self.transacted_value = Decimal(0.0)
        self.cost_of_transaction = Decimal(0.0)

def process_order(pnl, oda):

    pos_delta = oda[0]
    mtm = oda[1]
            
    pnl.prev_units = pnl.cum_units
    pnl.cum_units += pos_delta

    fees = Decimal(0.0)

    trade_in_same_direction = sign(pnl.prev_units) == sign(pnl.cum_units)
    increase_position = abs(pnl.cum_units) > abs(pnl.prev_units)

    pnl.transacted_value = pos_delta * pnl.cost_basis - fees

    if pnl.prev_units == 0:
        pnl.cost_basis = mtm #Decimal(0)
    elif trade_in_same_direction: 
        if increase_position:
            pnl.cost_basis = (pnl.cost_basis * abs(pnl.prev_units) + mtm * abs(pos_delta)) / abs(pnl.cum_units)
        # else: #do nothing
    else:
        pnl.cost_basis = mtm

    pnl.cost_of_transaction = pos_delta * mtm + fees

    if abs(pnl.cum_units) < abs(pnl.prev_units):
        pnl.realised_pnl =  pnl.transacted_value - pnl.cost_of_transaction
    else:
        pnl.realised_pnl = 0
   
    pnl.potential_pnl = (pnl.cost_basis - mtm) * pnl.cum_units - fees

    #print("cost basis {}, cost_of_trans = {} pnl.transacted_value {} pnl.cum_units {} prev_units {} mtm {} trade_in_same_direction {}".format(pnl.cost_basis,pnl.cost_of_transaction, pnl.transacted_value, pnl.cum_units, pnl.prev_units, mtm, trade_in_same_direction))

def walkforward(args=None):
    args = parse_args(args)

    # prices (from influxdb)
    # ... we'll sample ladders and use ~10 levels of depth to calculate mark-to-market
    prices = pd.read_csv(
        args.data0, 
        converters={'Date': int, 'Open': decimal.Decimal},
        usecols = ['Date', 'Open'],
     )
    prices = prices.set_index('Date')

    # trades (from pg_executions)
    trades = pd.read_csv(
        args.data1, 
        converters={'datetime': int, 'price': decimal.Decimal, 'quantity': decimal.Decimal},
        usecols = ['datetime', 'quantity', 'price', 'product'],
     )
    trades = trades.set_index('datetime')
       
    # funding (generally from pg_balance, but need historical data, not just most recent)
    if os.path.isfile(args.data2):
        funding = pd.read_csv(
            args.data2, 
            converters={'value': decimal.Decimal, 'created_at': pgtime_to_epoch},
            usecols = ['created_at', 'value', 'currency'],
        )

        funding['created_at'] = pd.to_datetime(funding['created_at'], format='%Y-%m-%d %H:%M:%S').astype(int)
        funding = funding.set_index('created_at')
    else:
        funding = pd.DataFrame()

    if args.opening_balance and not args.opening_balance.is_nan():
        funding_ccy = 'JPY' 
        #TODO handle difference currencies
        #TODO convert to portfolio currency

        data_dict = [{
            'created_at':iso8601_to_epoch(args.fromdate),
            'value':Decimal(args.opening_balance), 
            'currency': funding_ccy
        }]

        df = pd.DataFrame.from_records(data_dict)
        df = df.set_index('created_at')

        funding = pd.concat([funding, df], sort = True)
    
    funding = funding[['value','currency']]

    if args.fromdate:
        fromtimestamp = iso8601_to_epoch(args.fromdate)
        prices = prices[prices.index.searchsorted(fromtimestamp):]
        trades = trades[trades.index.searchsorted(fromtimestamp):]
        funding = funding[funding.index.searchsorted(fromtimestamp):]

    if args.todate:
        totimestamp = iso8601_to_epoch(args.todate)
        prices = prices[:prices.index.searchsorted(totimestamp)]
        trades = trades[:trades.index.searchsorted(totimestamp)]
        funding = funding[:funding.index.searchsorted(totimestamp)]

    dit = peekable(prices.iterrows())
    oit = peekable(trades.iterrows())
    fit = peekable(funding.iterrows())
   
    end = [sys.maxsize,0,0,5]
 
    data = []
    last_roi = []

    pnl = pnl_tracker()

    last_dt = 0
    px_dt = dit.peek(end)[0]
    od_dt = oit.peek(end)[0]
    fd_dt = fit.peek(end)[0]

    realised_pnl = Decimal(0)
    potential_pnl = Decimal(0)
    balance = Decimal(0)

    mtm = Decimal('Nan')

    event_type = []

    while px_dt != end[0] or od_dt != end[0] or fd_dt != end[0]:
        
        if px_dt <= od_dt and px_dt <= fd_dt: # process market data tick
            event_type = ['tick']
            last_dt = px_dt
            md = next(dit)[1]
            px_dt = dit.peek(end)[0]

            if pnl.cum_units != 0.0:
                mtm = md[0]
                
                potential_pnl = (mtm - pnl.cost_basis) * pnl.cum_units

                cum_pnl = realised_pnl + potential_pnl

                if logging.DEBUG >= logging.root.level:                                                                                                                                                                                                                                                                             
                    print("mkt {} px {} adj_pnl {} cum_pnl {} cum_units {:f} prev_units {} cost_of_transaction {} cum_cost {} transacted_value {}".format(
                        last_dt, 
                        round_sigfig(mtm,6), 
                        round_sigfig(pnl.potential_pnl, 3),
                        round_sigfig(cum_pnl, 3),
                        round_sigfig(pnl.cum_units,3), 
                        round_sigfig(pnl.prev_units), 
                        round_sigfig(pnl.cost_of_transaction), 
                        round_sigfig(pnl.cum_cost), 
                        round_sigfig(pnl.transacted_value),
                        )
                    )
        elif od_dt <= fd_dt: # process trade

            event_type = ['orda']
            noda = next(oit)[1]
            last_dt = od_dt
            od_dt = oit.peek(end)[0] 
            
            pos_delta = noda[0]
            assert(pos_delta != Decimal(0))

            mtm = noda[1]
            assert(mtm != Decimal(0))

            trade_in_same_direction = sign(pnl.cum_units + pos_delta) == sign(pnl.cum_units)

            if trade_in_same_direction:
                process_order(pnl, noda)
 
                realised_pnl += pnl.realised_pnl
                potential_pnl = pnl.potential_pnl
            else: # net out

                oda_copy = noda.copy()

                # TODO consider reflecting closed position as cashflow
                units_closed = -pnl.cum_units
                units_left = pos_delta + pnl.cum_units

                oda_copy[0] = units_closed
                process_order(pnl, oda_copy)
                realised_pnl += pnl.realised_pnl

                oda_copy[0] = units_left 
                process_order(pnl, oda_copy)
                assert(pnl.realised_pnl == Decimal(0))
                potential_pnl = pnl.potential_pnl

            cum_pnl = realised_pnl + potential_pnl
 
            if logging.DEBUG >= logging.getLogger().getEffectiveLevel(): 
                print("oda {} px {} adj_pnl {:f} cum_pnl {} cum_units {:f} real {} potential {} prev_units {:f} cost_of_transaction {} cum_cost {} transacted_value {} pos_delta {:f}".format(
                    last_dt, 
                    round_sigfig(mtm,6),
                    round_sigfig(cum_pnl / abs(pnl.cum_units)) if pnl.cum_units != 0 else 0,
                    round_sigfig(cum_pnl, 3),
                    round_sigfig(pnl.cum_units,3),
                    realised_pnl,
                    potential_pnl,
                    round_sigfig(pnl.prev_units,3), 
                    round_sigfig(pnl.cost_of_transaction), 
                    round_sigfig(pnl.cum_cost), 
                    round_sigfig(pnl.transacted_value),
                    round_sigfig(pos_delta,3),
                    )
                )
             
        else: # process cashflow
            event_type = ['xfer']
            # Rn = (Equity at the end of the sub-period – Equity at the beginning of the sub-period – Deposits + Withdrawals) / Equity at the beginning of the sub-period

            fd = next(fit)[1]
            last_dt = fd_dt

            balance += fd[0]
            fd_dt = fit.peek(end)[0]

        next_dt = min(px_dt, od_dt, fd_dt)

        if last_dt < next_dt: # close of timepoint

            account_value = balance + realised_pnl + potential_pnl

            roi = [realised_pnl, potential_pnl, balance, account_value]

            if logging.DEBUG >= logging.getLogger().getEffectiveLevel(): 
                print("roi {} next_roi {} balance {} unrealised {}".format(roi, last_roi, balance, Decimal(abs(pnl.cum_units) * mtm)))
            
            if roi != last_roi or event_type == ['orda']:
                data.append([last_dt, mtm, pnl.cost_basis, pnl.cum_units] + roi)
                last_roi = [] + roi #induce copy

            last_dt = next_dt
    
        #         r_est = (emv - bmv - balance_adj)/(bmv + balance_adj)
        #         print (bmv + balance_adj*(1 + r_est))
        #         r_est = (emv - bmv - balance_adj*(1+r_est) - balance_adj*(1+r_est)*r_est) / (bmv + balance_adj*(1 + r_est))
        #         r_est = (emv - bmv - balance_adj*(1+r_est) - balance_adj*(1+r_est)*r_est) / (bmv + balance_adj*(1 + r_est))
        #         r_est = (emv - bmv - balance_adj*(1+r_est) - balance_adj*(1+r_est)*r_est) / (bmv + balance_adj*(1 + r_est))

        #         print("cash {} r_est {} bmv {} emv {}".format(last_dt, r_est*100, bmv, emv))

    if args.writer: 
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, 'w', newline='') as csvfile:
            pnlwriter = csv.writer(csvfile, delimiter='\t', 
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            pnlwriter.writerow(["Date", "Price", "Cost Basis", "NOP", "Realised Pnl", "Potential Pnl", "Balance", "Account Value"])
            for x in data:
                pnlwriter.writerow(x)
        print("wrote report {}".format(csvfile))

    return data[-1]

    # perf = df['Returns'].calc_stats() 
    # perf.display() 
    # print(perf)   
    # perf.plot()  
            

def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            'Walk-forward historical simulation'
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
    parser.add_argument('--data1', default='sample/trades.csv',
                        required=False, help='Data to read in')

    # select 
    #     *
    # from 
    #     import_from_quoine_main.pg_balances 
    # where 
    #     user_id = 401130    
    parser.add_argument('--data2', default='sample/pg_balances.in.csv',
                        required=False, help='Data to read in')

    # Defaults for dates
    parser.add_argument('--fromdate', required=False, default='',
                        help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')

    parser.add_argument('--todate', required=False, default='',
                        help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')

    parser.add_argument('--plot', required=False, default='',
                        nargs='?', const='{}',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--opening_balance', type=Decimal, required=False, default='Nan',
                        help='Real number')

    # parser.add_argument('--timeframe', default='minutes', required=False,
    #                     choices=['ticks', 'microseconds', 'seconds',
    #                              'minutes', 'daily', 'weekly', 'monthly'],
    #                     help='Timeframe to resample to')

    # parser.add_argument('--compression', default=1, required=False, type=int,
    #                     help=('Compress n bars into 1'))

    parser.add_argument('--writer', required=False, action='store_true',
                        help=('Add a Writer'))

    parser.add_argument('--csv', default='output/report.csv', required=False,
                        help=('Output to csv'))

    return parser.parse_args(pargs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)

    walkforward()



