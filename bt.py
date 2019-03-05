#!/usr/bin/env python3
# -*- coding: utf-8; py-indent-offset:4 -*-
"""Simulate trading performance with reference to external trade history. 

Starting from a historical snapshot of account value and open positions, we walk forward in time and replay market events to reconstruct the trader's pnl profile.
https://en.wikipedia.org/wiki/Walk_forward_optimization

FIXME correct mark-to-market (mtm) by using price by size
FIXME convert pnl to portfolio currency using correct fx conversion rate
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

sign = lambda x: math.copysign(1, x)
def round_sigfig(x, sig=5):
    if x == 0: 
        return 0 
    elif x.is_nan():
        return Decimal('Nan')
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


class perf_ohlc:
    def __init__(   self):
        self.units_open = Decimal(0.0)
        self.units_high = Decimal(0.0)
        self.units_low = Decimal(0.0)
        self.units_close = Decimal(0.0)
        self.pnl_open = Decimal(0.0)
        self.pnl_high = Decimal(0.0)
        self.pnl_low = Decimal(0.0)
        self.pnl_close = Decimal(0.0)
        self.px_open = Decimal('Nan')
        self.px_high = Decimal('Nan')
        self.px_low = Decimal('Nan')
        self.px_close = Decimal('Nan')
        self.trade_cnt = 0


def init_perf_ohlc(ohlc, units_val, pnl_val, px_val):
    ohlc.units_open = units_val
    ohlc.units_high = units_val
    ohlc.units_low = units_val
    ohlc.units_close = units_val
    ohlc.pnl_open = pnl_val
    ohlc.pnl_high = pnl_val
    ohlc.pnl_low = pnl_val
    ohlc.pnl_close = pnl_val
    ohlc.px_open = px_val
    ohlc.px_high = px_val
    ohlc.px_low = px_val
    ohlc.px_close = px_val
    ohlc.trade_cnt = 0

def update_perf_ohlc(ohlc, units_val, pnl_val, px_val):
    if ohlc.units_close != units_val:
        ohlc.trade_cnt = ohlc.trade_cnt + 1
    ohlc.units_high = max(ohlc.units_high, units_val)
    ohlc.units_low = min(ohlc.units_low, units_val)
    ohlc.units_close = units_val
    ohlc.pnl_high = max(ohlc.pnl_high, pnl_val)
    ohlc.pnl_low = min(ohlc.pnl_low, pnl_val)
    ohlc.pnl_close = pnl_val
    ohlc.px_high = max(ohlc.px_high, px_val)
    ohlc.px_low = min(ohlc.px_low, px_val)
    ohlc.px_close = px_val

def finish_perf_ohlc(ohlc):
    ohlc.trade_cnt = 0
    ohlc.units_open = ohlc.units_close
    ohlc.units_high = ohlc.units_close
    ohlc.units_low = ohlc.units_close
    ohlc.pnl_open = ohlc.pnl_close
    ohlc.pnl_high = ohlc.pnl_close
    ohlc.pnl_low = ohlc.pnl_close
    ohlc.px_open = ohlc.px_close
    ohlc.px_high = ohlc.px_close
    ohlc.px_low = ohlc.px_close


class pnl_tracker:
    def __init__(   self):
        self.mtm = Decimal('Nan')
        self.potential_pnl = Decimal(0.0)
        self.cum_cost = Decimal(0.0)
        self.cost_basis = Decimal('Nan')
        self.realised_pnl = Decimal(0.0)
        self.prev_units = Decimal(0.0)
        self.cum_units = Decimal(0.0)
        self.transacted_value = Decimal(0.0)
        self.cost_of_transaction = Decimal(0.0)
        self.cum_pnl = Decimal(0.0)
        self.unit_ccy = str()
        self.pnl_ccy = str()

def process_order(pnl, oda):

    pos_delta = oda[0]
    pnl.mtm = oda[1]
    assert(pnl.mtm != Decimal(0) and not pnl.mtm.is_nan())

    pnl.prev_units = pnl.cum_units
    pnl.cum_units += pos_delta
    pnl.cum_units = pnl.cum_units.normalize()

    fees = Decimal(0.0)

    trade_in_same_direction = sign(pnl.prev_units) == sign(pnl.cum_units)
    increase_position = abs(pnl.cum_units) > abs(pnl.prev_units)

    pnl.transacted_value = pos_delta * pnl.cost_basis - fees

    if pnl.prev_units == 0:
        pnl.cost_basis = pnl.mtm #Decimal(0)
    elif trade_in_same_direction: 
        if increase_position:
            pnl.cost_basis = (pnl.cost_basis * abs(pnl.prev_units) + pnl.mtm * abs(pos_delta)) / abs(pnl.cum_units)
        # else: #do nothing
    else:
        pnl.cost_basis = pnl.mtm

    pnl.cost_of_transaction = pos_delta * pnl.mtm + fees

    if abs(pnl.cum_units) < abs(pnl.prev_units):
        pnl.realised_pnl += pnl.transacted_value - pnl.cost_of_transaction
    else:
        pnl.realised_pnl += 0
   
    pnl.potential_pnl = (pnl.cost_basis - pnl.mtm) * pnl.cum_units - fees

    pnl.cum_pnl = pnl.realised_pnl + pnl.potential_pnl

    if pnl.cum_units == Decimal(0):
        pnl.cost_basis = Decimal('Nan')
 
    #print("cost basis {}, cost_of_trans = {} pnl.transacted_value {} pnl.cum_units {} prev_units {} mtm {} trade_in_same_direction {}".format(pnl.cost_basis,pnl.cost_of_transaction, pnl.transacted_value, pnl.cum_units, pnl.prev_units, mtm, trade_in_same_direction))

def walkforward(args=None):
    args = parse_args(args)

    # FIXME make data driven
    portfolio_ccy = args.portfolio_ccy

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

    # trades (from pg_executions)
    trades = pd.read_csv(
        args.data1, 
        converters={'datetime': int, 'price': decimal.Decimal, 'quantity': decimal.Decimal},
        usecols = ['datetime', 'quantity', 'price', 'product']
        #user_id', 'quantity', 'price', 'product', 'product_type', 'base_ccy', 'quoted_ccy'],
     )
    trades = trades.set_index('datetime')
       
    #trades = trades[['quantity','price','product','product_type','base_ccy','quoted_ccy']]

    # funding (generally from pg_balance, but need historical data, not just most recent)
    if os.path.isfile(args.data2):
        funding = pd.read_csv(
            args.data2, 
            converters={'value': decimal.Decimal, 'created_at': pgtime_to_epoch},
            usecols = ['created_at', 'currency', 'value'],
        )

        funding['created_at'] = pd.to_datetime(funding['created_at'], format='%Y-%m-%d %H:%M:%S').astype(int)
        funding = funding.set_index('created_at')
    else:
        funding = pd.DataFrame()

    if args.opening_balance:
        for funding_ccy, quantity in args.opening_balance.items():
            data_dict = [{
                'created_at': iso8601_to_epoch(args.fromdate),
                'currency': funding_ccy,
                'value': Decimal(quantity), 
            }]

            df = pd.DataFrame.from_records(data_dict)
            df = df.set_index('created_at')

            funding = pd.concat([funding, df], sort = True)
    
    funding = funding[['currency','value']]

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
    balance_data = []
    last_roi = []

    #first_dt = 0
    last_dt = 0
    px_dt = dit.peek(end)[0]
    od_dt = oit.peek(end)[0]
    fd_dt = fit.peek(end)[0]

    asset = dict()

    product_perf = dict()
    product_ohlc = dict()
    portfolio_open = Decimal(0)
    portfolio_high = Decimal(0)
    portfolio_low = Decimal(0)
    portfolio_close = Decimal(0)

    last_bar = 0

    while px_dt != end[0] or od_dt != end[0] or fd_dt != end[0]:
        
        if px_dt <= od_dt and px_dt <= fd_dt: # process market data tick
            # event_type = ['tick']
            last_dt = px_dt

            md = next(dit)[1]
            px_dt = dit.peek(end)[0]
                        
            product_id = md[1]  
            assert (product_id)

            if product_id in product_perf:
       
                p_pnl = product_perf[product_id]
                if p_pnl.cum_units != 0:
                    #FIXME improve accuracy by looking up price by size market data feed
                    #methodology must be consistent with walk backward
                    p_pnl.mtm = md[0]
                        
                    if p_pnl.pnl_ccy == portfolio_ccy:
                        portfolio_close -= p_pnl.cum_pnl
                    else:
                        # FIXME
                        # use fx conversion rate to convert p_pnl.cum_pnl from p_pnl.pnl_ccy to portfolio_ccy
                        fx_conv_rate = Decimal(1.0)
                        portfolio_close -= fx_conv_rate * p_pnl.cum_pnl

                    p_pnl.potential_pnl = (p_pnl.mtm - p_pnl.cost_basis) * p_pnl.cum_units
                    p_pnl.cum_pnl = p_pnl.realised_pnl + p_pnl.potential_pnl             

                    assert(product_id in product_ohlc)
                    # ohlc created upon opening the position
                    #  have been created when the position 
                    ohlc = product_ohlc[product_id]
                    update_perf_ohlc(ohlc, p_pnl.cum_units, p_pnl.cum_pnl, p_pnl.mtm)
                    
                    if p_pnl.pnl_ccy == portfolio_ccy:
                        portfolio_close += p_pnl.cum_pnl
                    else:
                        # FIXME
                        # use fx conversion rate to convert p_pnl.cum_pnl from p_pnl.pnl_ccy to portfolio_ccy
                        fx_conv_rate = Decimal(1.0)
                        portfolio_close += fx_conv_rate * p_pnl.cum_pnl

                    portfolio_low = min(portfolio_low, portfolio_close)
                    portfolio_high = max(portfolio_high, portfolio_close)

                    if logging.DEBUG >= logging.root.level:                                                                                                                                                                                                                                                                             
                        print("mkt {} px {} adj_pnl {} cum_pnl {} cum_units {:f} prev_units {} cost_of_transaction {} cum_cost {} transacted_value {}".format(
                            last_dt, 
                            round_sigfig(p_pnl.mtm, 6), 
                            round_sigfig(p_pnl.potential_pnl, 3),
                            round_sigfig(p_pnl.cum_pnl, 3),
                            round_sigfig(p_pnl.cum_units, 3), 
                            round_sigfig(p_pnl.prev_units), 
                            round_sigfig(p_pnl.cost_of_transaction), 
                            round_sigfig(p_pnl.cum_cost), 
                            round_sigfig(p_pnl.transacted_value),
                            )
                        )

        elif od_dt <= fd_dt: # process trade
            # event_type = ['orda']
            last_dt = od_dt

            noda = next(oit)[1]
            od_dt = oit.peek(end)[0] 
            
            pos_delta = noda[0]
            assert(pos_delta != Decimal(0))

            product_id = noda[2]
            assert (product_id)

            if product_id in product_perf:                
                p_pnl = product_perf[product_id]
            else:
                product_perf[product_id] = pnl_tracker()
                p_pnl = product_perf[product_id]
                p_pnl.unit_ccy = product_static[str(product_id)]['base']
                p_pnl.pnl_ccy = product_static[str(product_id)]['quoted']

            trade_in_same_direction = sign(p_pnl.cum_units + pos_delta) == sign(p_pnl.cum_units)
 
            if p_pnl.pnl_ccy == portfolio_ccy:
                portfolio_close -= p_pnl.cum_pnl
            else:
                # FIXME
                # use fx conversion rate to convert p_pnl.cum_pnl from p_pnl.pnl_ccy to portfolio_ccy
                fx_conv_rate = Decimal(1.0)
                portfolio_close -= fx_conv_rate * p_pnl.cum_pnl


            if trade_in_same_direction:
                process_order(p_pnl, noda)

            else: # net out

                oda_copy = noda.copy()

                # TODO consider reflecting closed position as cashflow
                units_closed = -p_pnl.cum_units
                units_left = pos_delta + p_pnl.cum_units

                oda_copy[0] = units_closed
                process_order(p_pnl, oda_copy)
               
                oda_copy[0] = units_left 
                process_order(p_pnl, oda_copy)

 
            if p_pnl.pnl_ccy == portfolio_ccy:
                portfolio_close += p_pnl.cum_pnl
            else:
                # FIXME
                # use fx conversion rate to convert p_pnl.cum_pnl from p_pnl.pnl_ccy to portfolio_ccy
                fx_conv_rate = Decimal(1.0)
                portfolio_close += fx_conv_rate * p_pnl.cum_pnl


            portfolio_low = min(portfolio_low, portfolio_close)
            portfolio_high = max(portfolio_high, portfolio_close)               

            if product_id in product_ohlc:
                ohlc = product_ohlc[product_id]
                update_perf_ohlc(ohlc, p_pnl.cum_units, p_pnl.cum_pnl, p_pnl.mtm)

            else:       
                product_ohlc[product_id] = perf_ohlc()
                ohlc = product_ohlc[product_id]
                init_perf_ohlc(ohlc, 0, 0, p_pnl.mtm)
                update_perf_ohlc(ohlc, p_pnl.cum_units, p_pnl.cum_pnl, p_pnl.mtm)
 

            if logging.DEBUG >= logging.getLogger().getEffectiveLevel(): 
                print("oda {} px {} adj_pnl {:f} cum_pnl {} cum_units {:f} real {} potential {} prev_units {:f} cost_of_transaction {} cum_cost {} transacted_value {} pos_delta {:f}".format(
                    last_dt, 
                    round_sigfig(p_pnl.mtm,6),
                    round_sigfig(p_pnl.cum_pnl / abs(p_pnl.cum_units)) if p_pnl.cum_units != 0 else 0,
                    round_sigfig(p_pnl.cum_pnl, 3),
                    round_sigfig(p_pnl.cum_units, 3),
                    p_pnl.realised_pnl,
                    p_pnl.potential_pnl,
                    round_sigfig(p_pnl.prev_units, 3), 
                    round_sigfig(p_pnl.cost_of_transaction), 
                    round_sigfig(p_pnl.cum_cost), 
                    round_sigfig(p_pnl.transacted_value),
                    round_sigfig(pos_delta, 3),
                    )
                )
             
        else: # process cashflow
            # event_type = ['xfer']
            # Rn = (Equity at the end of the sub-period – Equity at the beginning of the sub-period – Deposits + Withdrawals) / Equity at the beginning of the sub-period

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

        if last_bar == 0:
            last_bar = int(last_dt//60)

        next_dt = min(px_dt, od_dt, fd_dt)

        next_bar = int(next_dt//60)

        if last_bar != next_bar: # close bar
            last_bar = next_bar

            roi = [asset, portfolio_open, portfolio_high, portfolio_low, portfolio_close]
            issame = (roi == last_roi) #only record entry if change occurred
            if not issame: 
                if logging.DEBUG >= logging.getLogger().getEffectiveLevel(): 
                    print("roi {} last_roi {}".format(roi, last_roi))
     
                trade_cnt = 0
                for product_id, ohlc in product_ohlc.items():
                    p_pnl = product_perf[product_id]
                    # FIXME
                    # convert from p_pnl.pnl_ccy to portfolio_ccy for ohlc values
                    #    ohlc.pnl_open, ohlc.pnl_high, ohlc.pnl_low, ohlc.pnl_close

                    trade_cnt += ohlc.trade_cnt
                    data.append([last_dt, product_id, ohlc.px_open, ohlc.px_high, ohlc.px_low, ohlc.px_close, p_pnl.cost_basis, p_pnl.unit_ccy, ohlc.units_open, ohlc.units_high, ohlc.units_low, ohlc.units_close, ohlc.trade_cnt, p_pnl.pnl_ccy, ohlc.pnl_open, ohlc.pnl_high, ohlc.pnl_low, ohlc.pnl_close])
                    finish_perf_ohlc(ohlc)

                data.append([last_dt, None, None, None, None, None, None, None, None, None, None, None, trade_cnt, portfolio_ccy, portfolio_open, portfolio_high, portfolio_low, portfolio_close])

                for product_id, p_pnl in product_perf.items():
                    if product_id in product_ohlc and p_pnl.cum_units == Decimal(0):
                        del product_ohlc[product_id]

                last_roi = [] + roi

                # for fairness we set close to the last open
                portfolio_open = portfolio_close

                # we must reset porfolio values 
                portfolio_high = portfolio_close
                portfolio_low = portfolio_close                
               

#    if logging.DEBUG >= logging.getLogger().getEffectiveLevel(): 
#        print("roi {} next_roi {} balance {} unrealised {}".format(roi, last_roi, balance, Decimal(abs(pnl.cum_units) * mtm)))
    
    if roi != last_roi:
        if logging.DEBUG >= logging.getLogger().getEffectiveLevel(): 
            print("roi {}".format(roi))
   
        trade_cnt = 0
        for product_id, ohlc in product_ohlc.items():
            p_pnl = product_perf[product_id]
            # FIXME
            # convert from p_pnl.pnl_ccy to portfolio_ccy for ohlc values
            #    ohlc.pnl_open, ohlc.pnl_high, ohlc.pnl_low, ohlc.pnl_close
            trade_cnt += ohlc.trade_cnt
            data.append([last_dt, product_id, ohlc.px_open, ohlc.px_high, ohlc.px_low, ohlc.px_close, p_pnl.cost_basis, p_pnl.unit_ccy, ohlc.units_open, ohlc.units_high, ohlc.units_low, ohlc.units_close, ohlc.trade_cnt, p_pnl.pnl_ccy, ohlc.pnl_open, ohlc.pnl_high, ohlc.pnl_low, ohlc.pnl_close])
            finish_perf_ohlc(ohlc)

        data.append([last_dt, None, None, None, None, None, None, None, None, None, None, None, trade_cnt, portfolio_ccy, portfolio_open, portfolio_high, portfolio_low, portfolio_close])

        for product_id, p_pnl in product_perf.items():
            if product_id in product_ohlc and p_pnl.cum_units == Decimal(0):
                del product_ohlc[product_id]
            

        last_roi = [] + roi

        # for fairness we set close to the last open
        portfolio_open = portfolio_close

        # we must reset porfolio values 
        portfolio_high = portfolio_close
        portfolio_low = portfolio_close 
    
        #         r_est = (emv - bmv - balance_adj)/(bmv + balance_adj)
        #         print (bmv + balance_adj*(1 + r_est))
        #         r_est = (emv - bmv - balance_adj*(1+r_est) - balance_adj*(1+r_est)*r_est) / (bmv + balance_adj*(1 + r_est))
        #         r_est = (emv - bmv - balance_adj*(1+r_est) - balance_adj*(1+r_est)*r_est) / (bmv + balance_adj*(1 + r_est))
        #         r_est = (emv - bmv - balance_adj*(1+r_est) - balance_adj*(1+r_est)*r_est) / (bmv + balance_adj*(1 + r_est))

        #         print("cash {} r_est {} bmv {} emv {}".format(last_dt, r_est*100, bmv, emv))

    if args.writer: 
        os.makedirs(os.path.dirname(args.report_csv), exist_ok=True)
        with open(args.report_csv, 'w', newline='') as csvfile:
            pnlwriter = csv.writer(csvfile, delimiter='\t', 
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            pnlwriter.writerow(["epoch", "product_id", "open_px", "high_px", "low_px", "close_px", "cost_basis", "unit_ccy", "open_nop", "high_nop", "low_nop", "close_nop", "trade_cnt", "pnl_ccy", "open_pnl", "high_pnl", "low_pnl", "close_pnl"])
 
            for x in data:
                pnlwriter.writerow(x)

        print("wrote report {}".format(csvfile))

        os.makedirs(os.path.dirname(args.balance_csv), exist_ok=True)
        with open(args.balance_csv, 'w', newline='') as csvfile:
            pnlwriter = csv.writer(csvfile, delimiter='\t', 
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            pnlwriter.writerow(["epoch", "ccy", "qty"])
 
            for x in balance_data:
                pnlwriter.writerow(x)

        print("wrote report {}".format(csvfile))

    return [asset.copy(), product_perf.copy()]

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
    # order by id asc
    # limit 200
    # ;
    parser.add_argument('--data1', default='sample/trades.in.csv',
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

    parser.add_argument('--opening_balance', type=json.loads, required=False,
                        help='Real number')

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

    parser.add_argument('--portfolio_ccy', default='JPY', required=False,
                        help=('Currency to value portfolio'))

    return parser.parse_args(pargs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)

    walkforward()



