import java.io.File;
import java.util.Locale;
import org.orekit.data.DataContext;
import org.orekit.data.DataSource;
import org.orekit.data.DirectoryCrawler;
import org.orekit.data.LazyLoadedDataContext;
import org.orekit.files.ccsds.ndm.ParserBuilder;
import org.orekit.files.sp3.SP3Parser;
import org.orekit.files.rinex.observation.RinexObservationParser;
import org.orekit.files.rinex.navigation.RinexNavigationParser;

/** Independent syntax acceptance, not trajectory/estimator qualification. */
public final class OrekitDataIOAcceptance {
    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        if (args.length != 3) throw new IllegalArgumentException("<data-dir> <family> <input>");
        LazyLoadedDataContext context = DataContext.getDefault();
        context.getDataProvidersManager().clearProviders();
        context.getDataProvidersManager().addProvider(new DirectoryCrawler(new File(args[0])));
        ParserBuilder builder = new ParserBuilder(context).withMu(3.986004418e14).withDefaultMass(100.0);
        DataSource source = new DataSource(new File(args[2]));
        int count;
        switch (args[1]) {
            case "oem":
                var oem = builder.buildOemParser().parseMessage(source);
                count = oem.getSegments().stream().mapToInt(s -> s.getCoordinates().size()).sum();
                System.out.println("first_x_m=" + oem.getSegments().get(0).getCoordinates().get(0).getPosition().getX());
                break;
            case "opm":
                var opm = builder.buildOpmParser().parseMessage(source);
                count = 1;
                System.out.println("first_x_m=" + opm.getPVCoordinates().getPosition().getX());
                break;
            case "omm":
                var omm = builder.buildOmmParser().parseMessage(source);
                count = 1;
                System.out.println("norad_id=" + omm.getData().getTLEBlock().getNoradID());
                break;
            case "aem":
                var aem = builder.buildAemParser().parseMessage(source);
                count = aem.getSegments().get(0).getData().getAngularCoordinates().size();
                var last = aem.getSegments().get(0).getData().getAngularCoordinates().get(count - 1);
                System.out.println("last_reference_x_body_y=" + last.getRotation().applyTo(org.hipparchus.geometry.euclidean.threed.Vector3D.PLUS_I).getY());
                System.out.println("last_body_rate_z=" + last.getRotationRate().getZ());
                break;
            case "apm":
                builder.buildApmParser().parseMessage(source);
                count = 1;
                break;
            case "tdm":
                var tdm = builder.buildTdmParser().parseMessage(source);
                count = tdm.getSegments().get(0).getData().getObservations().size();
                break;
            case "cdm":
                builder.buildCdmParser().parseMessage(source);
                count = 2;
                break;
            case "sp3":
                var sp3 = new SP3Parser().parse(source);
                count = sp3.getSatellites().size();
                break;
            case "rinex-observation":
                count = new RinexObservationParser().parse(source).getObservationDataSets().size();
                break;
            case "rinex-navigation":
                new RinexNavigationParser().parse(source);
                count = 1;
                break;
            default: throw new IllegalArgumentException("Unsupported family " + args[1]);
        }
        System.out.println("parsed_records=" + count);
    }
}
