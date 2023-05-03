import pandas as pd

class Backtest:
    def __init__(self, data, strategy):
        self.data = data
        self.strategy = strategy
    
    def run_backtest(self):
        # Initialize variables
        position = 0
        entry_price = 0
        exit_price = 0
        pnl = []
        
        for i in range(len(self.data)):
            # Check if there is an open position
            if position == 0:
                signal = self.strategy.generate_signal(self.data.iloc[i])
                if signal == 1:
                    position = 1
                    entry_price = self.data['close'][i]
                elif signal == -1:
                    position = -1
                    entry_price = self.data['close'][i]
            elif position == 1:
                exit_signal = self.strategy.generate_exit_signal(self.data.iloc[i])
                if exit_signal == 1:
                    position = 0
                    exit_price = self.data['close'][i]
                    pnl.append(exit_price - entry_price)
            elif position == -1:
                exit_signal = self.strategy.generate_exit_signal(self.data.iloc[i])
                if exit_signal == -1:
                    position = 0
                    exit_price = self.data['close'][i]
                    pnl.append(entry_price - exit_price)
        
        return pnl
