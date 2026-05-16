import re
import numpy as np

class DateTokenizer:
    def __init__(self):
        self.days = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        self.months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        self.leaps = ['True', 'False']
        self.decades = [str(i) for i in range(180, 221)] 
        
        self.date_chars = ['<PAD>', '<SOS>', '<EOS>', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.char2idx = {char: idx for idx, char in enumerate(self.date_chars)}
        self.idx2char = {idx: char for idx, char in enumerate(self.date_chars)}

    def parse_line(self, line):
        conditions = re.findall(r'\[(.*?)\]', line)
        date_str = line.split(']')[-1].strip()
        return conditions, date_str

    def encode_conditions(self, conditions):
        day, month, leap, decade = conditions
        return [
            self.days.index(day),
            self.months.index(month),
            self.leaps.index(leap),
            self.decades.index(decade)
        ]   

    def encode_date(self, date_str):
        if not date_str:
            return []
            
        d, m, y = date_str.split('-')
        reordered_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}" 
        
        sequence = ['<SOS>'] + list(reordered_date) + ['<EOS>']
        return [self.char2idx[char] for char in sequence]

    def decode_date(self, token_ids):
        chars = []
        for idx in token_ids:
            char = self.idx2char.get(idx, '')
            if char == '<EOS>':
                break
            if char not in ['<SOS>', '<PAD>']:
                chars.append(char)
                
        reordered_date = "".join(chars)
        try:
            y, m, d = reordered_date.split('-')
            return f"{int(d)}-{int(m)}-{y}"
        except ValueError:
            return reordered_date
            
    def process_dataset(self, filepath):
        X, y = [], []
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip(): continue
                conditions, date_str = self.parse_line(line)
                X.append(self.encode_conditions(conditions))
                if date_str:
                    y.append(self.encode_date(date_str))
                    
        return np.array(X), np.array(y)