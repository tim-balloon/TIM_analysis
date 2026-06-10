from astroplan import Observer, FixedTarget, Constraint, AltitudeConstraint
from astroplan import is_observable, is_always_observable, observability_table
from astroplan.plots import plot_altitude, plot_finder_image
from astropy.coordinates import Angle, SkyCoord, EarthLocation, get_body
from astropy.time import Time
import astropy.units as u
from astropy.visualization import time_support, quantity_support
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from IPython import embed
import pickle
from astropy.table import Table
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u

time_support()
quantity_support()

EL_MIN = 20
EL_MAX = 52
DAZ_MIN = 90
DAZ_MAX = 225

LDB = (-77.861 * u.deg, 167.061 * u.deg)
FLOAT_ALT = 37000 * u.m

def wrap360(ang): return ((ang + 360.) % 360.)

class SunRelativeAzConstraint(Constraint):
    '''
    Constrain an asymmetrical sun-relative azimuth pointing of the boresight to
    target.
    These angles are the angles that the boresight must point away from the sun
    in order to maintain the instrument and electronics in the shade of the
    sunshade. As always, azimuth is measured +clockwise from the reference
    point, in this case the sun azimuth. The range is [0, 360] deg.
    '''
    def __init__(self, min=None, max=None):
        '''
        min : `~astropy.units.Quantity` or `None` (optional)
            Minimum acceptable azimuth angle between sun and boresight. `None`
            indicates no limit.
        max : `~astropy.units.Quantity` or `None` (optional)
            Maximum acceptable azimuth angle between sun and boresight. `None`
            indicates no limit.
        '''
        self.min = min if min is not None else 0*u.deg
        self.max = max if max is not None else 360*u.deg


    def compute_constraint(self, times, observer, targets):
        # targets = get_skycoord(targets)
        sun_body = get_body('sun', times, location=observer.location)
        sun = SkyCoord(ra=sun_body.ra, dec=sun_body.dec, obstime=times,
                       location=observer.location)
        sun_altaz = sun.transform_to('altaz')

        # Calculate separation between boresight and sun
        # Targets are automatically converted to SkyCoord objects
        # by __call__ before compute_constraint is called.

        try:
            n_targets = len(targets)
        except TypeError as e:
            targets = SkyCoord([targets,])
            n_targets = len(targets)
        try:
            n_times = len(times)
        except TypeError as e:
            # times = [times,]
            n_times = len(times)

        delta_az = np.atleast_2d(np.empty((n_targets, n_times)) * u.deg)
        for i, target in enumerate(targets):
            target_with_loc = SkyCoord(ra=target.ra, dec=target.dec, obstime=times, location=observer.location)
            target_altaz = target_with_loc.transform_to('altaz')
            delta_az[i,:] = wrap360(target_altaz.az.deg - sun_altaz.az.deg) * u.deg # time axis
        # print(n_targets, n_times, delta_az.shape, delta_az)

        if self.min is None and self.max is not None:
            mask = self.max >= delta_az
        elif self.max is None and self.min is not None:
            mask = self.min <= delta_az
        elif self.min is not None and self.max is not None:
            mask = ((self.min <= delta_az) &
                    (delta_az <= self.max))
        else:
            raise ValueError("No max and/or min specified in "
                             "SunRelativeAzConstraint.")
        return mask

def get_observer(launch_lat, launch_lon, float_alt, times):
    '''Generate an observer object with lat/lon that change over time like a balloon.'''
    t = (times - times[0]).to(u.hr)
    ldb = (launch_lat, launch_lon)
    # ground track:
    # Salter Test Flight Universal completed ~1 circuit in 11 days, 6 hr, 57 min.
    # Average the longitudinal velocity. Really, ballons have a spatial velocity,
    # so this constant lat assumption is an oversimplification.
    dlon_dt = -(360 * u.deg / (11 * u.day + 6 * u.hr + 57 * u.min)).to(u.deg / u.hr)
    lon = ldb[1] + dlon_dt * t
    lon = wrap360(lon.to(u.deg).value) * u.deg
    # add a little lat wobble, eyeballed from STFU flight track
    lat = 0 * u.deg + np.ones_like(lon.value) * ldb[0] + .2 * u.deg * np.sin(lon.to(u.rad) * 15)
    observer = Observer(longitude=lon, latitude=lat, elevation=float_alt, name="TIM")
    return observer

def observability(targets:list, observer:Observer, times, plot=True):
    constraints = [
        AltitudeConstraint(EL_MIN * u.deg, EL_MAX * u.deg),
        SunRelativeAzConstraint(min=DAZ_MIN * u.deg, max=DAZ_MAX * u.deg)
    ]

    table = observability_table(constraints, observer, targets, times)

    if plot:
        # https://astroplan.readthedocs.io/en/latest/tutorials/constraints.html
        for j, target in enumerate(targets):
            observability_grid = np.zeros((len(constraints), len(times)))
            for i, constraint in enumerate(constraints):
                # Evaluate each constraint
                observability_grid[i, :] = constraint(
                    observer,
                    target,
                    times=times,
                    grid_times_targets=True
                )

            # Create plot showing observability of the target:
            extent = [-0.5, -0.5+len(times), -0.5, len(constraints) - 0.5]

            fig, ax = plt.subplots(figsize=(10,4))
            ax.imshow(observability_grid, extent=extent, cmap='bone_r', vmin=0, vmax=1, origin='lower')

            ax.set_yticks(range(0, len(constraints)))
            ax.set_yticklabels([c.__class__.__name__ for c in constraints])

            ax.set_xticks(range(len(times)))
            ax.set_xticklabels([t.datetime.strftime("%H:%M") for t in times])

            ax.set_xticks(np.arange(extent[0], extent[1]), minor=True)
            ax.set_yticks(np.arange(extent[2], extent[3]), minor=True)

            ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
            ax.tick_params(axis='x', which='minor', bottom='off')
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

            ax.tick_params(axis='y', which='minor', left='off')
            ax.set_xlabel('Time on {0} (UTC)'.format(times[0].datetime.date()))
            fig.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)
            ax.set_title(f'{target.name}: {table["fraction of time observable"][j] * (times[-1] - times[0]).to(u.hr):.2f}' + 
                         '\nBlack = Observable')
            fig.tight_layout()
        plt.show()

    return table

def time_vs_altitude(targets:list, observer, times):
    fig, ax = plt.subplots(figsize=(12,4))
    from astroplan.plots import plot_altitude, plot_finder_image

    plot_altitude(targets, observer, times, ax=ax)
    #ax.axhline(EL_MIN, color='limegreen')
    #ax.axhline(EL_MAX, color='limegreen')
    #ax.axhspan(EL_MIN, EL_MAX, color='limegreen', alpha=0.3)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    # Extract the plotted curves
    curves = {}
    for line in ax.lines:
        label = line.get_label()
        xdata = np.asarray(line.get_xdata())  # time values (matplotlib dates)
        ydata = np.asarray(line.get_ydata())  # altitude values in degrees
        mask = (ydata > EL_MIN) & (ydata < EL_MAX)
        curves[label] = {'times':xdata, 'altitude':ydata, 'mask':mask}
        #plt.show()
    return fig, ax, curves

def time_vs_sun_relative_az(targets:list, observer, times):
    '''
    Plot the sun-relative azimuth of the observer when observing the target.
    This is a delta azimuth, target az - sun az. Positive angles denote the
    observer is pointed to a greater azimuth angle than the sun,
    clockwise viewed from above.
    '''
    fig, ax = plt.subplots(figsize=(12,4))
    for target in targets:
        ax.plot(times,wrap360(observer.altaz(times, target=target).az.deg -observer.sun_altaz(times).az.deg),marker='.',label=target.name)
    #ax.axhline(DAZ_MIN, color='limegreen')
    #ax.axhline(DAZ_MAX, color='limegreen')
    #ax.axhspan(DAZ_MIN, DAZ_MAX, color='limegreen', alpha=0.3)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    #plt.show()
    # Extract the plotted curves
    curves = {}
    for line in ax.lines:
        xdata = line.get_xdata()  # time values (matplotlib dates)
        x = [t.datetime for t in xdata]
        label = line.get_label()
        ydata = np.asarray(line.get_ydata())  # altitude values in degrees
        mask = (ydata > DAZ_MIN) & (ydata < DAZ_MAX)
        curves[label] = {'times': x, 'sun_relative_az':ydata, 'mask':mask}
        #plt.show()
    return fig, ax, curves

def ground_track(observer, times):
    this_lon = observer.longitude
    this_lat = observer.latitude
    t = (times - times[0]).to(u.hr)
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(projection='polar'))
    mappable = ax.scatter(this_lon.to(u.rad), this_lat, c=range(len(this_lon)))
    ax.set_rmin(-90)
    ax.set_rmax(-70)
    # ax.set_rticks(np.arange(-90, -70, 5))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(360)
    ax.grid(True)
    ax.set_title(f'Looking Down:\nGround Track: {(t[-1] - t[0]).to(u.day).value:.1f} days')
    fig.colorbar(mappable, ax=ax, label='Time Since Launch (hr)')
    fig.tight_layout()
    ax.set_facecolor('lightgrey')
    fig.set_facecolor('lightgrey')
    plt.show()
    return fig, ax

# Define GOODS-S once outside the loop
GOODS_S_COORD = SkyCoord(ra='3h32m36.51s', dec='-27d47m33.74s')
def get_goodss_unobservable_mask(observer, times):
    """Returns a boolean mask: True where GOODS-S is NOT observable."""
    constraints = [AltitudeConstraint(EL_MIN * u.deg, EL_MAX * u.deg),  SunRelativeAzConstraint(min=DAZ_MIN * u.deg, max=DAZ_MAX * u.deg)    ]
    goods_s = FixedTarget(SkyCoord(ra=GOODS_S_COORD.ra, dec=GOODS_S_COORD.dec,obstime=times, location=observer.location),name='GOODS-S')
    masks = np.array([c(observer, goods_s, times=times, grid_times_targets=True)[0]
                      for c in constraints])
    goodss_observable = np.all(masks, axis=0) # True where GOODS-S IS observable
    return ~goodss_observable                 # True where GOODS-S is NOT observable

if __name__ == '__main__':

    #-----------------------------------------------------------------
    data = Table.read('reuter_2020_data/smg_dndz_tables_in_TIM.fits', hdu=1)
    data.sort('z_spec')
    ra_list, dec_list = data['RA'], data['DEC']
    target_names = data['spt_id']
    #--------------------------------------------------------------------

    #--------------------------------------------------------------------
    launch_location = EarthLocation(lat=LDB[0], lon=LDB[1], height=FLOAT_ALT)
    start = datetime(2027, 12, 1, 0, 0, 0)
    end   = datetime(2028, 1, 31, 0, 0, 0)   # or whatever end date you want
    #--------------------------------------------------------------------

    step_hours = 5
    step = timedelta(hours=step_hours)  # 10h steps
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d %H:%M:%S"))
        current += step
    times = Time(dates, scale='utc', location=launch_location) #'2026-12-25 00:00:00'
    tim = get_observer(launch_location.lat,launch_location.lon,launch_location.height,times )
    targets = []
    goods_s = FixedTarget(SkyCoord(ra='3h32m36.51s',dec='-27d47m33.74s',obstime=times,location=tim.location ), name='GOODS-S')
    targets += [goods_s,]
    for ismg, (name, ra, dec) in enumerate(zip(target_names,ra_list, dec_list)):
        target = FixedTarget(SkyCoord(ra=ra*u.degree,dec=dec*u.degree,obstime=times,location=tim.location), name=f'{name}' )
        targets += [target,]

    if(True): _, _, curves_alt = time_vs_altitude(targets, tim, times)
    if(True): _, _, curves_sun = time_vs_sun_relative_az(targets, tim, times)
    plt.close('all')

    fig, (ax3, ax1, ax2) = plt.subplots(1,3, figsize=(15,5), dpi=200)
    curve = 'GOODS-S'
    ax1.set_ylabel('altitude')
    ax2.set_ylabel('relative distance to sun')
    ax1.set_xlabel('Date')
    ax2.set_xlabel('Date')
    ax3.set_ylabel('DEC [deg]')
    ax3.set_xlabel('RA [deg]')
    combined_mask_GOODS_S = ~(curves_alt[curve]['mask'] & curves_sun[curve]['mask'])
    print(f"GOODS-S observable at {curves_alt['GOODS-S']['mask'].sum()} timesteps")
    print(f"GOODS-S NOT observable at {combined_mask_GOODS_S.sum()} timesteps")
    print(f"Total timesteps: {len(combined_mask_GOODS_S)}")
    ax3.scatter(GOODS_S_COORD.ra.deg, GOODS_S_COORD.dec.deg, 
           marker='*', s=300, color='k', label='GOODS-S', zorder=5)


    ax1.axhspan(EL_MIN, EL_MAX, color='grey', alpha=0.1)
    ax2.axhspan(DAZ_MIN, DAZ_MAX, color='grey', alpha=0.1)
    # Highlight dates where GOODS-S IS observable (i.e. ~combined_mask_GOODS_S)
    goodss_times = np.asarray(curves_sun['GOODS-S']['times'])
    goodss_obs_mask = ~combined_mask_GOODS_S  # True where GOODS-S IS observable
    for t in goodss_times[goodss_obs_mask]:
        ax1.axvspan(t, t + timedelta(hours=step_hours), color='gold', alpha=0.3, lw=0)
        ax2.axvspan(t, t + timedelta(hours=step_hours), color='gold', alpha=0.3, lw=0)
    # Add a legend entry for the highlight
    from matplotlib.patches import Patch
    gold_patch = Patch(color='gold', alpha=0.3, label='GOODS-S')
    colors = cm.coolwarm(np.linspace(0.,1,len(curves_alt)))

    for igl, (name, c) in enumerate(zip(target_names, colors)):
        time = data['T'][igl,:] 
        if(np.any(time<1)): curves_alt[name]['line_mask'] = True
        else: curves_alt[name]['line_mask'] = False
        curves_alt[name]['z'] = data['z_spec'][igl]
        curves_alt[name]['ra'] = data['RA'][igl]
        curves_alt[name]['dec'] = data['DEC'][igl]
                    
    for curve, c in zip(curves_alt, colors):
        if('SPT' in curve):

            times = np.asarray(curves_sun[curve]['times'])
            total_time = len(times) * step_hours
            combined_mask = curves_alt[curve]['mask'] & curves_sun[curve]['mask'] & combined_mask_GOODS_S
            w = np.where(combined_mask)
            observable_time = len(w[0]) * step_hours

            if(len(w[0]) == 0): 
                print("'"+curve+"'", ':', False, ',') 
                continue
                
            if(not curves_alt[curve]['line_mask']):
                print("'"+curve+"'", ':', False, ',') 
                continue
                

            #w = np.where(curves_alt[curve]['mask'] & curves_sun[curve]['mask'])
            ax1.plot(times[w], curves_alt[curve]['altitude'][w],'.',c=c)
            ax2.plot(times[w], curves_sun[curve]['sun_relative_az'][w],'.', label=f'{curve}\n{observable_time:.0f}h,z={curves_alt[curve]["z"]:.1f}',c=c)
            ax3.scatter(curves_alt[curve]['ra'], curves_alt[curve]['dec'],color=c)
            print("'"+curve+"'", ':', True, ',') 
        if('GOOD' in curve):
            '''
            times = np.asarray(curves_sun[curve]['times'])
            combined_mask = ~combined_mask_GOODS_S
            w = np.where(combined_mask)
            if(len(w[0]) == 0): continue
            #w = np.where(curves_alt[curve]['mask'] & curves_sun[curve]['mask'])
            ax1.plot(times[w], curves_alt[curve]['altitude'][w], '-ok')
            ax2.plot(times[w], curves_sun[curve]['sun_relative_az'][w], '-ok',label=curve, )
            '''

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    handles, labels = ax2.get_legend_handles_labels()
    handles.append(gold_patch)
    ax2.legend(handles=handles, bbox_to_anchor=(1.1,1.1), ncol=2)

    fig.tight_layout()
    plt.show()

    if(False):
        step = timedelta(days=4)
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d %H:%M:%S"))
            current += step

        list_fraction_osbervable_time = []
        listfraction = [] 
        for d in dates:
            launch_time = Time(d, scale='utc', location=launch_location) #'2026-12-25 00:00:00'
            step_hr = 3.0 * u.hr
            my_duration = 24 * 3 * u.hr
            timespan = np.arange(0, my_duration.value + step_hr.value, step_hr.value) * u.hr
            times = launch_time + timespan
            tim = get_observer(launch_location.lat,launch_location.lon,launch_location.height,times )
            targets = []
            for ismg, (name, ra, dec) in enumerate(zip(target_names,ra_list, dec_list)):
                target = FixedTarget(SkyCoord(ra=ra*u.degree,dec=dec*u.degree,obstime=times,location=tim.location), name=f'{name}' )
                targets += [target,]
            table = observability(targets,tim,times,plot=False)
            #print(table)
            list_fraction_osbervable_time.append( table["fraction of time observable"] )

            targets = []
            goods_s = FixedTarget(SkyCoord(ra='3h32m36.51s',dec='-27d47m33.74s',obstime=times,location=tim.location ), name='GOODS-S')
            targets += [goods_s,]
            table = observability(targets,tim,times,plot=False)
            listfraction.append( table["fraction of time observable"] ) 

        
        if(False): fig, ax = time_vs_altitude(targets, tim, times)
        if(False): fig, ax = time_vs_sun_relative_az(targets, tim, times)
        list_fraction_osbervable_time = np.asarray(list_fraction_osbervable_time)
        listfraction = np.asarray(listfraction)
        colors = cm.coolwarm(np.linspace(0.,1,len(target_names)))

        plt.figure(figsize=(20,10))
        for igl, (name, c) in enumerate(zip(target_names, colors)):
            # convert string dates to datetime
            date_objects = [datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in dates]
            y = list_fraction_osbervable_time[:,igl] 
            time = data['T'][igl,:] 
            
            if(np.any(time<1) and (np.any(y*my_duration.value))>=1):
                plt.plot(date_objects, y, marker="o",color=c, label = f'{name}, z={data["z_spec"][igl]:.1f}',)
            else: plt.plot(date_objects, y, marker="o",color=c, label = f'{name}', alpha=0.2)
        plt.plot(date_objects, listfraction[:,0], marker="o",color='k', label = f'GOODS-S')
        # format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.xlabel("Date")
        plt.ylabel("Fraction of observable time\nover a period of 3 days")
        plt.legend(bbox_to_anchor = (1,1), title='SPT ID', ncol=3)
        plt.tight_layout()
        
    

    
    '''
    
    list_fraction_osbervable_time = []
    listfraction = [] 
    for d in dates:
        launch_time = Time(d, scale='utc', location=launch_location)
        step_hr = 3.0 * u.hr
        my_duration = 24 * 3 * u.hr
        timespan = np.arange(0, my_duration.value + step_hr.value, step_hr.value) * u.hr
        times = launch_time + timespan
        tim = get_observer(launch_location.lat, launch_location.lon, launch_location.height, times)
        # --- NEW: mask out times when GOODS-S is observable ---
        goodss_not_obs_mask = get_goodss_unobservable_mask(tim, times)
        times_excl_goodss = times[goodss_not_obs_mask]
        # If ALL times are excluded, append zeros and skip
        if len(times_excl_goodss) == 0:
            list_fraction_osbervable_time.append(np.zeros(len(target_names)))
            listfraction.append([0.0])
            continue
        # Recompute observer for the filtered times
        tim_excl = get_observer(launch_location.lat, launch_location.lon,launch_location.height, times_excl_goodss)
        # --- SMG targets on filtered times ---
        targets = []
        for name, ra, dec in zip(target_names, ra_list, dec_list):
            target = FixedTarget(SkyCoord(ra=ra*u.degree, dec=dec*u.degree,obstime=times_excl_goodss, location=tim_excl.location),name=f'{name}')
            targets.append(target)
        table = observability(targets, tim_excl, times_excl_goodss, plot=False)
        list_fraction_osbervable_time.append(table["fraction of time observable"])
        # --- GOODS-S fraction on filtered times (should be ~0 by construction) ---
        goods_s = FixedTarget(  SkyCoord(ra=GOODS_S_COORD.ra, dec=GOODS_S_COORD.dec,obstime=times_excl_goodss, location=tim_excl.location),name='GOODS-S')
        table_gs = observability([goods_s], tim_excl, times_excl_goodss, plot=False)
        listfraction.append(table_gs["fraction of time observable"])



    list_fraction_osbervable_time = np.asarray(list_fraction_osbervable_time)
    listfraction = np.asarray(listfraction)
    colors = cm.coolwarm(np.linspace(0.,1,len(target_names)))

    plt.figure(figsize=(20,10))
    
    for igl, (name, c) in enumerate(zip(target_names, colors)):
        # convert string dates to datetime
        date_objects = [datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in dates]
        y = list_fraction_osbervable_time[:,igl] 
        time = data['T'][igl,:] 
        
        if(np.any(time<1) and (np.any(y*my_duration.value))>=1):
            plt.plot(date_objects, y, marker="o",color=c, label = f'{name}, z={data["z_spec"][igl]:.1f}',)
        else: plt.plot(date_objects, y, marker="o",color=c, label = f'{name}', alpha=0.2)
    
    plt.plot(date_objects, listfraction[:,0], marker="o",color='k', label = f'GOODS-S')
    # format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.xlabel("Date")
    plt.ylabel("Fraction of observable time\nover a period of 3 days")
    plt.legend(bbox_to_anchor = (1,1), title='SPT ID', ncol=3)
    plt.tight_layout()
    plt.show()

    '''