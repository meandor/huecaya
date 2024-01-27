from pyspark.sql import SparkSession
from de_identification_library.udf import de_identify

SPARK = SparkSession.builder.getOrCreate()


def test_apply_udf():
    dataframe = SPARK.createDataFrame([["Peter Lustig text"]], ["text"])

    actual = dataframe.withColumn("de_identified_text", de_identify("text")).drop(
        "text"
    )
    expected = SPARK.createDataFrame([["xxx text"]], ["de_identified_text"])

    assert expected.collect() == actual.collect()
