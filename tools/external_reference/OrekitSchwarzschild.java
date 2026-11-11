import java.io.*;
import java.util.Locale;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;
import org.orekit.forces.gravity.Relativity;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.*;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.numerical.NumericalPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.PVCoordinates;

/** Stock Orekit force and orbit references; no OEL equations used. */
public final class OrekitSchwarzschild {
    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        double mu = 3.986004418e14;
        AbsoluteDate epoch = AbsoluteDate.J2000_EPOCH;
        Relativity force = new Relativity(mu);
        try(PrintWriter fixed = new PrintWriter(new FileWriter(args[0]));
            PrintWriter trajectory = new PrintWriter(new FileWriter(args[1]))) {
            fixed.println("case,x,y,z,vx,vy,vz,ax,ay,az");
            trajectory.println("case,time_s,x,y,z,vx,vy,vz");
            double[][] states = {{7000000,0,0,0,6000,4500}, {42164000,0,0,0,3074,0},
                {7000000,0,0,1200,9000,1000}, {-15000000,23000000,7000000,-2200,-1300,3000},
                {0,0,7000000,7500,500,100}, {7000000,0,0,0,0,0}};
            for(int i=0;i<states.length;i++) {
                double[] s=states[i];
                CartesianOrbit orbit=new CartesianOrbit(new PVCoordinates(new Vector3D(s[0],s[1],s[2]),
                    new Vector3D(s[3],s[4],s[5])),FramesFactory.getGCRF(),epoch,mu);
                SpacecraftState state=new SpacecraftState(orbit);
                Vector3D a=force.acceleration(state,force.getParameters(epoch));
                fixed.print(i);
                for(double value:s) fixed.printf(",%.17g",value);
                fixed.printf(",%.17g,%.17g,%.17g%n",a.getX(),a.getY(),a.getZ());
                if(i>=3) continue;
                for(boolean enabled:new boolean[]{false,true}) {
                    double[][] tol=NumericalPropagator.tolerances(1e-6,orbit,OrbitType.CARTESIAN);
                    NumericalPropagator prop=new NumericalPropagator(new DormandPrince853Integrator(.001,120,tol[0],tol[1]));
                    prop.setOrbitType(OrbitType.CARTESIAN);
                    prop.setInitialState(state);
                    if(enabled) prop.addForceModel(force);
                    for(int t=0;t<=21600;t+=300) {
                        PVCoordinates pv=prop.propagate(epoch.shiftedBy(t)).getPVCoordinates();
                        trajectory.printf("%d_%s,%d",i,enabled?"relativity":"baseline",t);
                        for(Vector3D v:new Vector3D[]{pv.getPosition(),pv.getVelocity()})
                            trajectory.printf(",%.17g,%.17g,%.17g",v.getX(),v.getY(),v.getZ());
                        trajectory.println();
                    }
                }
            }
        }
    }
}
