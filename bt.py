#!/usr/bin/env python3
# -*- coding: utf-8; py-indent-offset:4 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from datetime import datetime
from decimal import Decimal
from more_itertools import peekable

import datetime
import decimal
import argparse
import calendar
import csv
import math
import pandas as pd
import sys

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

class pnl_tracker:
    def __init__(   self):
        self.potential_pnl = Decimal(0.0)
        self.buy = 0
        self.sell = 0
        self.cum_cost = Decimal(0.0)
        self.realised_pnl = Decimal(0.0)
        self.prev_units = Decimal(0.0)
        self.cum_units = Decimal(0.0)
        self.transacted_value = Decimal(0.0)
        self.previous_cost = Decimal(0.0)
        self.cost_of_transaction = 0.0

def process_order(pnl, oda):

    pos_delta = oda[0]
    mtm = oda[1]
            
    pnl.prev_units = pnl.cum_units

    fees = Decimal(0.0)
    pnl.transacted_value = abs(pos_delta) * mtm - fees
    pnl.previous_cost = Decimal(pnl.cum_cost)

    pnl.cum_units += pos_delta

    if pnl.prev_units != 0:
        pnl.cost_of_transaction = abs(pos_delta / pnl.prev_units * pnl.previous_cost)
    else:
        pnl.cost_of_transaction = 0

    if abs(pnl.cum_units) < abs(pnl.prev_units):
        pnl.cum_cost = pnl.previous_cost - pnl.cost_of_transaction
    else:
        pnl.cum_cost = pnl.previous_cost + abs(pnl.transacted_value)

    if pnl.cost_of_transaction == 0 or abs(pnl.cum_units) > abs(pnl.prev_units):
        pnl.realised_pnl = -fees
    else:
        if pos_delta < 0:
            pnl.realised_pnl = pnl.transacted_value - pnl.cost_of_transaction - fees
        else:
            pnl.realised_pnl = pnl.cost_of_transaction - pnl.transacted_value - fees           

    if pnl.cum_units > 0 and pnl.cost_of_transaction > 0:
        pnl.potential_pnl = mtm * pnl.cum_units - pnl.cum_cost - pnl.realised_pnl
    else:
        pnl.potential_pnl = 0

def runstrat(args=None):
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
       
    # prices (from pg_balance, but need historical data, not just most recent)
    funding = pd.read_csv(
        args.data2, 
        converters={'value': decimal.Decimal, 'created_at': pgtime_to_epoch},
        usecols = ['created_at', 'value'],
     )
    funding['created_at'] = pd.to_datetime(funding['created_at'], format='%Y-%m-%d %H:%M:%S').astype(int)
    funding = funding.set_index('created_at')

    realised_pnl = Decimal(0)
    potential_pnl = Decimal(0)

    dit = peekable(prices.iterrows())
    oit = peekable(trades.iterrows())
    fit = peekable(funding.iterrows())
   
    end = [sys.maxsize,0,0,5]

    noda = oit.peek(end)[1]
 
    data = []
    roi = []

    pnl = pnl_tracker()

    sign = lambda x: math.copysign(1, x)

    last_dt = 0
    px_dt = dit.peek(end)[0]
    od_dt = oit.peek(end)[0]
    fd_dt = fit.peek(end)[0]

    #maybe_new_roi_period = False

    balance = Decimal(0)

    #bmv = Decimal('Nan')
    #emv = Decimal('Nan')
    mtm = Decimal('Nan')
    min_nop = Decimal(0)
    max_nop = Decimal(0)

    while px_dt != end[0] or od_dt != end[0] or fd_dt != end[0]:
        
        if px_dt <= od_dt and px_dt <= fd_dt: # process market data tick
            last_dt = px_dt
            md = next(dit)[1]
            px_dt = dit.peek(end)[0]

            if pnl.cum_units != 0.0:
                mtm = md[0]
                
                value = abs(pnl.cum_units) * mtm
                potential_pnl = value - abs(pnl.cum_cost)

                cum_pnl = realised_pnl + potential_pnl
                                                                                                                                                                                                                                                                                                        
                print("mkt {} px {} adj_pnl {} cum_pnl {} cum_units {:f} prev_units {} cost_of_transaction {} cum_cost {} previous_cost {} transacted_value {}".format(
                    last_dt, 
                    round_sigfig(mtm,6), 
                    round_sigfig(cum_pnl / abs(pnl.cum_units),3) if pnl.cum_units != 0 else 0,
                    round_sigfig(cum_pnl, 3),
                    round_sigfig(pnl.cum_units,3), 
                    round_sigfig(pnl.prev_units), 
                    round_sigfig(pnl.cost_of_transaction), 
                    round_sigfig(pnl.cum_cost), 
                    round_sigfig(pnl.previous_cost), 
                    round_sigfig(pnl.transacted_value),
                    )
                )
        elif od_dt <= fd_dt: # process trade

            noda = next(oit)[1]
            last_dt = od_dt
            od_dt = oit.peek(end)[0] 
            
            pos_delta = noda[0]
            mtm = noda[1]

            if (sign(pnl.cum_units + pos_delta) == sign(pnl.cum_units)): # remain long or short

                #if not bmv.is_nan():
                if abs(pnl.cum_units + pos_delta) > abs(pnl.cum_units): # increase pos size
                    balance -= abs(pos_delta * mtm) # treat as deposit
                else: # decrease pos size
                    balance += abs(pos_delta * mtm) # treat as withdrawal

                process_order(pnl, noda)
                realised_pnl += pnl.realised_pnl
                potential_pnl = pnl.potential_pnl
            else: # net out
                # maybe_new_roi_period = True

                oda_copy = noda.copy()

                # TODO consider reflecting closed position as cashflow
                units_closed = -pnl.cum_units
                units_left = pos_delta + pnl.cum_units

                #if not bmv.is_nan():
                # withdrawal
                balance += abs(units_closed * mtm)

                # deposit
                balance -= abs(units_left * mtm)

                oda_copy[0] = units_closed
                process_order(pnl, oda_copy)
                realised_pnl += pnl.realised_pnl

                oda_copy[0] = units_left 
                process_order(pnl, oda_copy)
                realised_pnl += pnl.realised_pnl
                potential_pnl = pnl.potential_pnl

            if (min_nop.is_nan()):
                min_nop = abs(pnl.cum_units)
            else: 
                min_nop = min(min_nop, abs(pnl.cum_units))

            if (max_nop.is_nan()):
                max_nop = abs(pnl.cum_units)
            else:
                max_nop = max(max_nop, abs(pnl.cum_units))
        
            cum_pnl = realised_pnl + potential_pnl
 
            # if bmv.is_nan():
            #     bmv = pnl.cum_cost

            print("oda {} px {} adj_pnl {:f} cum_pnl {} cum_units {:f} real {} potential {} prev_units {:f} cost_of_transaction {} cum_cost {} previous_cost {} transacted_value {} pos_delta {:f}".format(
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
                round_sigfig(pnl.previous_cost), 
                round_sigfig(pnl.transacted_value),
                round_sigfig(pos_delta,3),
                )
            )
             
        else: # process cashflow

            # Rn = (Equity at the end of the sub-period – Equity at the beginning of the sub-period – Deposits + Withdrawals) / Equity at the beginning of the sub-period
            
            # equity_start = abs(pnl.cum_units)
            fd = next(fit)[1]
            last_dt = fd_dt
            # if bmv.is_nan():
            #     maybe_new_roi_period = True

            balance += fd[0] 
            fd_dt = fit.peek(end)[0]

        next_dt = min(px_dt, od_dt, fd_dt)

        if last_dt < next_dt: # close of timepoint

            if pnl.cum_units == 0:
                account_value = balance
            else:
                account_value = balance + Decimal(abs(pnl.cum_units) * mtm)

            next_roi = [min_nop, max_nop, realised_pnl + potential_pnl, account_value]
            if roi != next_roi:
                data.append([last_dt] + next_roi)
                roi = [] + next_roi

                min_nop = Decimal(0)
                max_nop = Decimal(0)

            last_dt = next_dt
        
        # if next_dt == end[0]:
        #     maybe_new_roi_period = True
        # if last_dt < next_dt and maybe_new_roi_period:
        #     maybe_new_roi_period = False
        #     if bmv.is_nan():
        #         if balance_adj.is_nan() or balance_adj.is_zero():
        #             bmv = Decimal(abs(pnl.cum_units) * mtm)
        #             balance_adj = Decimal(0)
        #         else:
        #             bmv = Decimal(balance_adj)
        #             balance_adj = Decimal(0)
        #     else:
        #         # close last bar
        #         print("close last bar {} px {} bmv {}".format(last_dt, mtm, bmv))
        #         # TODO aggregate on same time point
                

        #         #if (bmv + balance_adj) != 0) or (bmv != 0 and new_roi_period):
        #             # new_roi_period = False
        #             # value = abs(pnl.cum_units) * mtm
        #             # potential_pnl = value - abs(pnl.cum_cost)
        #         if not emv.is_nan():
        #             bmv = emv

        #         emv = pnl.cum_cost + balance_adj

        #         #emv = realised_pnl + Decimal(abs(pnl.cum_units) * mtm) + balance_adj
        #         print ("cum_units {}".format(abs(pnl.cum_units)))
        #         print ("balance_adj {}".format(bmv + balance_adj))
        #         print ("emv {}".format(bmv + emv))
        #         r_est = (emv - bmv - balance_adj)/(bmv + balance_adj)
        #         print (bmv + balance_adj*(1 + r_est))
        #         r_est = (emv - bmv - balance_adj*(1+r_est) - balance_adj*(1+r_est)*r_est) / (bmv + balance_adj*(1 + r_est))
        #         r_est = (emv - bmv - balance_adj*(1+r_est) - balance_adj*(1+r_est)*r_est) / (bmv + balance_adj*(1 + r_est))
        #         r_est = (emv - bmv - balance_adj*(1+r_est) - balance_adj*(1+r_est)*r_est) / (bmv + balance_adj*(1 + r_est))

        #         print("cash {} r_est {} bmv {} emv {}".format(last_dt, r_est*100, bmv, emv))
                

    if args.writer: 
        with open(args.csv, 'w', newline='') as csvfile:
            pnlwriter = csv.writer(csvfile, delimiter='\t', 
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            pnlwriter.writerow(["Date", "Min NOP", "Max NOP", "Pnl", "Balance"])
            for x in data:
                pnlwriter.writerow(x)

    # perf = df['Returns'].calc_stats() 
    # perf.display() 
    # print(perf)   
    # perf.plot()  
            


def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            'Order History Sample'
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

    parser.add_argument('--order-history', required=False, action='store_true',
                        help='use order history')

    parser.add_argument('--cerebro', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--broker', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--sizer', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--strat', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--plot', required=False, default='',
                        nargs='?', const='{}',
                        metavar='kwargs', help='kwargs in key=value format')

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
    runstrat()



