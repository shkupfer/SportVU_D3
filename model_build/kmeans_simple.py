import django
from django.db.models import Q, Case, When, IntegerField
from django.db.models import F as dF

django.setup()

from nbad3.models import Coords, Moment, Possession
from datetime import timedelta
import numpy as np
from ingest.utils import ball_team
import argparse
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from scipy import stats
from itertools import combinations
import sys

position_sort_order = ['', 'G', 'G-F', 'F-G', 'F', 'F-C', 'C-F', 'C']
examine_hc_secs = 6
try_n_clusters = range(4, 16 + 1)
# try_n_clusters = range(6, 12 + 1)

bad_games = [21500022, 21500514, 21500009, 21500552, 21500620, 21500004, 21500034, 21500623, 21500584, 21500639, 21500645, 21500628, 21500053, 21500616, 21500506, 21500560, 21500597, 21500621, 21500084, 21500519, 21500544, 21500582, 21500647, 21500002, 21500049, 21500539, 21500029, 21500542, 21500625, 21500520, 21500060, 21500023, 21500016, 21500500, 21500521, 21500011, 21500043, 21500557, 21500073, 21500536, 21500572, 21500574, 21500660, 21500013, 21500545, 21500026, 21500564, 21500567, 21500634, 21500012, 21500032, 21500561, 21500062, 21500513, 21500490, 21500019, 21500530, 21500657, 21500044, 21500507, 21500538, 21500633, 21500031, 21500081, 21500629, 21500594, 21500027, 21500071, 21500566, 21500532, 21500653, 21500525, 21500510, 21500061, 21500017, 21500528, 21500041, 21500553, 21500001, 21500517, 21500556, 21500066, 21500496, 21500048, 21500565, 21500003, 21500495, 21500580, 21500632, 21500039, 21500056, 21500571, 21500546, 21500618, 21500036, 21500033, 21500522, 21500569, 21500585, 21500648, 21500491, 21500658, 21500577, 21500493, 21500636]

right_basket_coords = np.array([89.25 - 47, 25])

scaler = 180 / np.pi


def angle_to_basket(coords, basket=right_basket_coords):
    x1 = basket[1] - coords[1]
    x2 = basket[0] - coords[0]
    theta = scaler * np.arctan2(x1, x2)
    return theta


def xy_dist_btwn_coords(c1, c2):
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

def centroid(coords_arr):
    x = np.mean([crd[0] for crd in coords_arr])
    y = np.mean([crd[1] for crd in coords_arr])
    return x, y


def run_kmeans(n, polar_extra, zscores, every, out_filename, cluster_method):
    print("Start of run_kmeans")
    all_data = []
    hc_possessions = Possession.objects.filter(valid=True, half_court=True).order_by('id')
    hc_possessions = hc_possessions.exclude(game__in=bad_games)
    if n:
        hc_possessions = hc_possessions[:n]
    total_poss = hc_possessions.count()
    print("Entering possessions loop")
    sys.stdout.flush()
    for ind, poss in enumerate(hc_possessions):
        if (ind + 1) % 10 == 0:
            print("Processing possession %s of %s" % (ind + 1, total_poss))

        hc_moments = (Moment.objects.filter(Q(game=poss.game)
                                            & Q(quarter=poss.start_event.period)
                                            & Q(game_clock__lte=poss.hc_start)
                                            & Q(game_clock__gte=poss.hc_start - timedelta(seconds=examine_hc_secs + .04)))
                      .distinct('real_timestamp', 'quarter', 'game_clock', 'shot_clock')).order_by('-game_clock')

        hc_moments = hc_moments[:25 * examine_hc_secs]
        if len(hc_moments) < 25 * examine_hc_secs:
            print("Less than 150 moments on this, skipping (%s):" % len(hc_moments))
            print(poss.long_desc())
            continue

        team_ball_case_stmt = Case(When(player_status__team=poss.team, then=1),
                                   When(player_status__team=ball_team, then=0),
                                   default=2, output_field=IntegerField())

        pos_case_stmt = Case(*[When(player_status__position=position, then=ind) for ind, position in enumerate(position_sort_order)],
                             output_field=IntegerField())

        first_mom_coords = Coords.objects.filter(moment=hc_moments[0])
        player_chans_translate = dict((pid, ind) for ind, pid in enumerate(first_mom_coords.order_by(team_ball_case_stmt, pos_case_stmt).values_list('player_status__id', flat=True)))


        all_coords = [list(Coords.objects.filter(moment=mom).values(x_new=dF('x') - 47 if poss.going_to_right else 47 - dF('x'))
                                                            .values_list('player_status__id', 'x_new', 'y')
                           ) for mom in hc_moments]

        ac_arr = np.array(all_coords)

        ps_ids = ac_arr[:, :, 0].flat
        try:
            ps_sort_order = [player_chans_translate[ps_id] for ps_id in ps_ids]
        except KeyError as exp:
            print("Couldn't find player status with ID %s. Skipping:" % str(exp))
            print(poss.long_desc())
            continue

        ac_arr[:, :, 0] = np.array(ps_sort_order).reshape(150, 11)

        poss_outputs = []
        for ts, mom_coords in enumerate(ac_arr):
            if not every or (ts % every == 0):
                srted = mom_coords[mom_coords[:, 0].argsort()]
                srted = srted[srted[:, 0] <= 5]
                out_coords = srted[:, 1:]
                if polar_extra:
                    cent_coords = centroid(out_coords[1:6])
                    out_coords = tuple(out_coords) + (right_basket_coords, cent_coords)
                    angles_to_basket = [angle_to_basket(crds) for crds in out_coords]

                    closest_to_ball = 0
                    min_dist_to_ball = None

                    for ind, ((n1, c1), (n2, c2)) in enumerate(combinations(enumerate(out_coords), 2)):
                        dist_btwn = xy_dist_btwn_coords(c1, c2)
                        if ind <= 4 and (min_dist_to_ball is None or dist_btwn < min_dist_to_ball):
                            closest_to_ball = ind
                            min_dist_to_ball = dist_btwn

                        angle_btwn = angles_to_basket[n2] - angles_to_basket[n1]
                        poss_outputs.extend([dist_btwn, angle_btwn])

                    poss_outputs.append(closest_to_ball)

                else:
                    poss_outputs.extend(out_coords.reshape(-1))

        all_data.append(tuple(poss_outputs))

        sys.stdout.flush()

    all_data = np.array(all_data)
    print("Output data shape: %s" % str(all_data.shape))

    if zscores:
        print("Getting zscores")
        for col in range(len(all_data[0])):
            all_data[:, col] = stats.zscore(all_data[:, col])

    if out_filename:
        print("Writing output to %s" % out_filename)
        np.save(out_filename, all_data)

    print("Starting clustering, using cluster method: %s" % str(cluster_method))
    for n_clusters in try_n_clusters:
        print("For %s clusters:" % n_clusters)
        clusterer = cluster_method(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(all_data)
        print("Calculating silhouette score")
        sil_score = silhouette_score(all_data, cluster_labels)
        print("Silhouette score: %s" % sil_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int)
    parser.add_argument('-polar_extra', '-p', action='store_true')
    parser.add_argument('-zscores', '-z', action='store_true')
    parser.add_argument('-every', '-e', type=int)
    parser.add_argument('-cluster_method', '-c', choices=['KM', 'AC', 'SC'], default='KM')
    parser.add_argument('-write_for_rnn', '-w')
    args = parser.parse_args()

    clusterers_dct = {'KM': KMeans, 'AC': AgglomerativeClustering, 'SC': SpectralClustering}
    cluster_method = clusterers_dct[args.cluster_method]

    run_kmeans(n=args.n, polar_extra=args.polar_extra, zscores=args.zscores, every=args.every,
               out_filename=args.write_for_rnn, cluster_method=cluster_method)
