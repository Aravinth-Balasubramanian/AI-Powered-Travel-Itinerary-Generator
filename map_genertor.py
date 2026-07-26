import folium


class MapGenerator:

    def __init__(self):
        pass

    def create_map(self, recommended_places, route_coordinates, output_file="travel_map.html"):

        # Create map centered on first place
        first_place = recommended_places[0]

        travel_map = folium.Map(
            location=[
                first_place["latitude"],
                first_place["longitude"]
            ],
            zoom_start=12
        )

        # Add markers
        for place in recommended_places:

            folium.Marker(
                location=[
                    place["latitude"],
                    place["longitude"]
                ],
                popup=place["name"],
                tooltip=place["name"],
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(travel_map)

        # Draw route
        if route_coordinates:

            folium.PolyLine(
                locations=[
                    [lat, lon] for lon, lat in route_coordinates
                ],
                color="blue",
                weight=5,
                opacity=0.8
            ).add_to(travel_map)

        # Save map
        travel_map.save(output_file)

        print(f"Map saved as {output_file}")

if __name__ == "__main__":

    recommended_places = [
        {
            "name": "Amber Fort",
            "latitude": 26.9855,
            "longitude": 75.8513
        },
        {
            "name": "Hawa Mahal",
            "latitude": 26.9239,
            "longitude": 75.8267
        }
    ]

    route_coordinates = [
        [75.8513, 26.9855],
        [75.8450, 26.9750],
        [75.8267, 26.9239]
    ]

    map_generator = MapGenerator()

    map_generator.create_map(
        recommended_places,
        route_coordinates
    )