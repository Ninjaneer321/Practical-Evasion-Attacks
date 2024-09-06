import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class NwkFeaturer:

    def __init__(self, **kwargs):

        # Default parameters values. If nI is not given, the code will crash later.
        params = {
            'folder': '../../nwk_data/',
            'network_file': 'train_scada-eth0',
            }
        params['network_results_path'] = params['folder'] + params['network_file'] + '.csv'

        for key,item in kwargs.items():
            params[key] = item

        self.params = params

    def get_command_time(self, a_mac, a_command, a_pd, addr_type):
        # Mask with the expression we need
        exp = (a_pd['Command'] == a_command) & (a_pd[str(addr_type) + ' src'] == a_mac)
        a_pd.loc[exp, 'Command Time'] = a_pd.loc[exp, 'Timestamp'] - a_pd.loc[exp, 'Timestamp'].shift(1)
        return a_pd
    
    def fill_command_field_column(self, pkts_pd, fillna=True, addr_type='IP'):
        unique_pkts = pkts_pd[~pkts_pd.duplicated(subset=[str(addr_type) + ' src', 'Command'], keep='first')]
        for i in range(len(unique_pkts.index)):
            try:
                pkts_pd = self.get_command_time(unique_pkts.iloc[i][str(addr_type) + ' src'],
                                                unique_pkts.iloc[i]['Command'], pkts_pd, addr_type)
            except ValueError:
                continue

        if fillna:
            pkts_pd['Command Time'] = pkts_pd['Command Time'].fillna(0)
        return pkts_pd

    def split_command_time_columns(self, pkt_pd):
        exp = ~pkt_pd.duplicated(subset=['IP src'], keep='first')
        ip_srcs_list = pkt_pd.loc[exp, :]['IP src'].to_list()
        ip_srcs_list.pop(0)

        for address in ip_srcs_list:
            exp = pkt_pd['IP src'] == address
            pkt_pd.loc[:, 'Ft_' + str(address)] = pkt_pd.loc[exp, 'Timestamp'] - pkt_pd.loc[exp, 'Timestamp'].shift(1)

        for address in ip_srcs_list:
            pkt_pd['Ft_' + str(address)] = pkt_pd['Ft_' + str(address)].fillna(0)

        return pkt_pd

    def fix_tcp_flags(self, tcp_packets):
        exp = tcp_packets['Command'].str.contains('0x00000001')
        tcp_packets.loc[exp, 'Command'] = 'FIN'

        exp = tcp_packets['Command'].str.contains('0x00000002')
        tcp_packets.loc[exp, 'Command'] = 'SYN'

        exp = tcp_packets['Command'].str.contains('0x00000004')
        tcp_packets.loc[exp, 'Command'] = 'RST'

        exp = tcp_packets['Command'].str.contains('0x00000008')
        tcp_packets.loc[exp, 'Command'] = 'PSH'

        exp = tcp_packets['Command'].str.contains('0x00000010')
        tcp_packets.loc[exp, 'Command'] = 'ACK'

        exp = tcp_packets['Command'].str.contains('0x00000011')
        tcp_packets.loc[exp, 'Command'] = 'FIN ACK'

        exp = tcp_packets['Command'].str.contains('0x00000012')
        tcp_packets.loc[exp, 'Command'] = 'SYN ACK'

        exp = tcp_packets['Command'].str.contains('0x00000014')
        tcp_packets.loc[exp, 'Command'] = 'RST ACK'

        exp = tcp_packets['Command'].str.contains('0x00000018')
        tcp_packets.loc[exp, 'Command'] = 'PSH ACK'

        return tcp_packets

    def basic_packet_engineering(self, pkts_pd):
        exp = pkts_pd['Protocol'] == 'ARP'
        pkts_pd.loc[exp, 'Command'] = pkts_pd.loc[exp, 'Info'].apply(
            lambda x: 'ARP Request' if "Who has" in x else 'ARP Response')

        exp = pkts_pd['Protocol'] == 'ENIP'
        pkts_pd.loc[exp, 'Command'] = pkts_pd.loc[exp, 'Info']

        exp = pkts_pd['Protocol'] == 'TCP'
        pkts_pd.loc[exp, 'Command'] = pkts_pd.loc[exp, 'Flags']

        exp = pkts_pd['Protocol'] == 'CIP CM'
        pkts_pd.loc[exp, 'Command'] = pkts_pd.loc[exp, 'Info'].str.extract(r'\-(.*?)\-')[0]

        exp = pkts_pd['Protocol'] == 'CIP'
        pkts_pd.loc[exp, 'Command'] = pkts_pd.loc[exp, 'Info'].str.extract(r'(.*?)\-')[0]

        # Flags is the same as Command now
        pkts_pd = pkts_pd.drop('Flags', axis=1)

        # Get command_time
        pkts_pd = self.fill_command_field_column(pkts_pd, fillna=False)
        return pkts_pd

    def packet_engineering(self, pkts_pd):
        # We process the ARP packets to obtain their command
        exp = pkts_pd['Protocol'] == 'ARP'
        pkts_pd.loc[exp, 'Command'] = pkts_pd.loc[exp, 'Info'].apply(
            lambda x: 'ARP Request' if "Who has" in x else 'ARP Response')

        exp = pkts_pd['Protocol'] == 'ENIP'
        pkts_pd.loc[exp, 'Command'] = pkts_pd.loc[exp, 'Info']

        exp = pkts_pd['Protocol'] == 'TCP'
        pkts_pd.loc[exp, 'Command'] = pkts_pd.loc[exp, 'Flags']

        exp = pkts_pd['Protocol'] == 'CIP CM'
        pkts_pd.loc[exp, 'Command'] = pkts_pd.loc[exp, 'Info'].str.extract(r'\-(.*?)\-')[0]

        exp = pkts_pd['Protocol'] == 'CIP'
        pkts_pd.loc[exp, 'Command'] = pkts_pd.loc[exp, 'Info'].str.extract(r'(.*?)\-')[0]

        # Flags is the same as Command now
        pkts_pd = pkts_pd.drop('Flags', axis=1)

        # Get command_time
        pkts_pd = self.fill_command_field_column(pkts_pd)

        # EXPERIMENTAL FEATURE
        pkts_pd = self.split_command_time_columns(pkts_pd)

        pkts_pd['Timestamp'] = pkts_pd['Timestamp'] - pkts_pd['Timestamp'].shift(1)
        pkts_pd['Timestamp'] = pkts_pd['Timestamp'].fillna(0)

        pkts_pd = pkts_pd.drop('Info', axis=1)

        # Timestamp is not necessary and HARD to reconstruct
        pkts_pd = pkts_pd.drop('Timestamp', axis=1)

        return pkts_pd

    def hot_encode(self, some_pkts):
        categorical_cols = [col for col
                            in some_pkts.columns if col not in ['Length', 'Command Time', 'Ft_192.168.1.2',
                                                                'Ft_192.168.1.1', 'Ft_10.0.2.1', 'Ft_10.0.3.1',
                                                                'Ft_10.0.4.1', 'Ft_10.0.5.1', 'Ft_10.0.6.1',
                                                                'Ft_10.0.7.1', 'Ft_10.0.8.1', 'Ft_10.0.9.1']]
        packets_categorical = pd.DataFrame(index=some_pkts.index, columns=categorical_cols,
                                               data=some_pkts[categorical_cols])
        packets_categorical = packets_categorical.fillna('None')

        enc = OneHotEncoder(handle_unknown='ignore')
        Z = enc.fit_transform(packets_categorical)

        # Create new dataframe with encoded categorical variables
        packets_encoded = pd.DataFrame(index=some_pkts.index,
                                       columns=enc.get_feature_names(packets_categorical.columns), data=Z.toarray())

        # Add continuous variables
        packets_encoded['Length'] = some_pkts['Length']
        packets_encoded['Command Time'] = some_pkts['Command Time']

        exp = ~some_pkts.duplicated(subset=['IP src'], keep='first')
        ip_srcs_list = some_pkts.loc[exp, :]['IP src'].to_list()
        ip_srcs_list.pop(0)

        for address in ip_srcs_list:
            packets_encoded['Ft_' + str(address)] = some_pkts['Ft_' + str(address)]

        return packets_encoded
